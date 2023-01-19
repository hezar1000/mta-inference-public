import argparse
import numpy as np
from collections import OrderedDict

from mta_inference.load_data import add_calibrations
from mta_inference.inference_utils import convertGradeScale25To5, buildAuthorGraph
from mta_inference.PG5_inference import run_gibbs
from mta_inference.database import create_session, ModelRun, decompress_data, save_samples

# from mta_inference.database import ModelRun, save_samples
# from mta_inference.database.utils import create_session
# from mta_inference.database.compression import decompress_data


HYPERPARAMS = [
    'mu_s', 'sigma_s',                               # true grades
    'sigma_tau', 'lambda_tau',                       # reliabilities
    'alpha_tau', 'beta_tau',                         # reliabilities (if correlation is disabled)
    'mu_b', 'sigma_b',                               # biases 
    'alpha_e', 'beta_e', 'tau_l',                    # efforts
    'p_uniform_high_effort', 'p_uniform_low_effort', # uniform mixing
    'disable_censoring',                             # censored/uncensored likelihood 
    'disable_biases', 'disable_efforts',             # skip updates
    'disable_correlation',                           # don't relate true grades and reliabilities
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Gibbs sampling for PG5.')
    # paths
    parser.add_argument('--database', type=str, required=True, help='Path to database')
    parser.add_argument('--model_run_id', required=True, help='Model run in database')
    parser.add_argument('--excluded_reviews', nargs='*', default=[], help='List of review IDs to exclude')
    
    # model ablations
    parser.add_argument('--disable_censoring', action='store_true', help='Use uncensored likelihood')
    parser.add_argument('--disable_biases', action='store_true', help='Clamp all biases to 0')
    parser.add_argument('--disable_efforts', action='store_true', help='Clamp all efforts to 1') 
    parser.add_argument('--disable_correlation', action='store_true', help="Don't relate true grades and reliabilities")

    # inference settings
    parser.add_argument('--grade_scale', type=str, required=True, help='Bin lookup to use for inference')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to record')
    parser.add_argument('--instructor_reliability', type=float, default=16, help='Instructor reliability for calibrations')
    parser.add_argument('--seed', type=int, help='Random seed for Gibbs sampler')
    parser.add_argument('--verbose', action='store_true', help='Print status while running inference')
    
    # hyperparams
    parser.add_argument('--mu_s',       type=float, required=True)
    parser.add_argument('--sigma_s',    type=float, required=True)
    parser.add_argument('--sigma_tau',  type=float, default=0)
    parser.add_argument('--lambda_tau', type=float, default=1)
    parser.add_argument('--alpha_tau',  type=float, default=0)
    parser.add_argument('--beta_tau',   type=float, default=0)
    parser.add_argument('--mu_b',       type=float, default=0)
    parser.add_argument('--sigma_b',    type=float, default=0.01)
    parser.add_argument('--alpha_e',    type=float, default=999) 
    parser.add_argument('--beta_e',     type=float, default=999) 
    parser.add_argument('--tau_l',      type=float, default=999)
    parser.add_argument('--p_uniform_high_effort', type=float, default=0.0)
    parser.add_argument('--p_uniform_low_effort', type=float, default=0.0)

    args = parser.parse_args()
    verboseprint = print if args.verbose else lambda *a, **k: None

    # check for bad arguments
    if args.disable_correlation:
        if args.alpha_tau == 0 or args.beta_tau == 0:
            verboseprint('Warning: correlation disabled, but missing value for alpha_tau or beta_tau!')
        if args.sigma_tau != 0 or args.lambda_tau != 1:
            verboseprint('Warning: correlation disabled, but got value for sigma_tau or lambda_tau!')
    else: # correlation enabled
        if args.sigma_tau == 0 or args.lambda_tau == 1:
            verboseprint('Warning: correlation enabled, but missing value for sigma_tau or lambda_tau!')
        if args.alpha_tau != 0 or args.beta_tau != 0:
            verboseprint('Warning: correlation enabled, but got value for alpha_tau or beta_tau!')
    
    # load data
    with create_session(args.database, verbose=args.verbose) as session:
        model_run = session.query(ModelRun).get(args.model_run_id)
        reported_grades, \
            graph, \
            ordered_graph, \
            calibration_grades, \
            effort_grades, \
            week_nums, \
            students, \
            submissions \
        = decompress_data(model_run.dataset.data)

    verboseprint('Setting up inference inputs...')

    # Build hyperparams
    args_dict = vars(args) 
    hyperparams = {hp: args_dict[hp] for hp in HYPERPARAMS}
    
    # Add calibration grades from instructor
    calibration_grades_no_dim = calibration_grades.sum(axis=1)
    reported_grades_instructor, graph_instructor, effort_grades_instructor = add_calibrations(
        reported_grades, graph, np.nan_to_num(calibration_grades), 1 * ~np.isnan(calibration_grades_no_dim), effort_grades,
    )

    # Clamp parameters
    reliabilities_clamped = np.full((len(students))+1, np.nan)
    biases_clamped = np.full((len(students))+1, np.nan)
    efforts_clamped = np.full((len(students))+1, np.nan)
    
    # - TA parameters and disabled parameters
    for i, (student, role) in enumerate(students.items()):
        if role == 'ta' or args.disable_biases: 
            biases_clamped[i] = 0.0
        if role == 'ta' or args.disable_efforts:
            efforts_clamped[i] = 1.0

    # - Instructor parameters
    reliabilities_clamped[-1] = args.instructor_reliability
    biases_clamped[-1] = 0.0
    efforts_clamped[-1] = 1.0

    clamped_values = {
        'reliabilities': reliabilities_clamped,
        'biases': biases_clamped,
        'efforts': efforts_clamped,
        'responsibilities' : effort_grades_instructor
    }

    # Rescale grades
    if args.grade_scale == '5':
        reported_grades_instructor = convertGradeScale25To5(reported_grades_instructor)

    #anonymize
    # students_anonymized = OrderedDict()
    # submissions_anonymized = OrderedDict()
    # for i, student_id in enumerate(students):
    #     students_anonymized[str(i+1)] = students[student_id]
    #     for sub_id, sub_info in submissions.items():
    #         if int(student_id) == int(sub_info[1]):
    #             submissions_anonymized[sub_id]= [sub_info[0], str(i+1)]


    # Create author graph
    author_graph = buildAuthorGraph(graph_instructor, submissions)

    # Run sampler
    verboseprint('Running inference...')
    if args.seed is not None:
        np.random.seed(args.seed)
    
    samples = run_gibbs(
        reported_grades_instructor,
        graph_instructor, 
        author_graph,
        hyperparams,
        clamped_values=clamped_values, 
        num_samples=args.num_samples,
        grade_scale=args.grade_scale,
        save_effort_draws=True,
        verbose=args.verbose,
    )

    # Add extra information
    samples['instructor_graph'] = graph_instructor
    sample_args = vars(args).copy()
    del sample_args['database']

    # Save
    verboseprint(f'Saving samples to database...')
    with create_session(args.database, verbose=args.verbose) as session:
        save_samples(session, model_run, samples, sample_args, commit=True)
