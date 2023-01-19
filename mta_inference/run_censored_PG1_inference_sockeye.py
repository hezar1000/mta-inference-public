import argparse
import numpy as np

from load_data import add_calibrations
import censored_PG1_inference

from mta_inference.database import create_session, ModelRun, decompress_data, save_samples


def convertGradeScale(grades):
    for (old_grade, new_grade) in [(25, 5), (20, 4), (16.25, 3), (12.5, 2), (6.25, 1)]:
        grades[np.where(grades == old_grade)] = new_grade

    return grades

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Gibbs sampling for censored PG1 model.')
    # paths
    parser.add_argument('--database', type=str, required=True, help='Path to database')
    parser.add_argument('--model_run_id', required=True, help='Model run in database')
    # parser.add_argument('--clamped_path', type=str, required=True, help='Location to read clamped values from') # TODO: zombies
    parser.add_argument('--excluded_reviews', nargs='*', default=[], help='List of review IDs to exclude')
    # inference settings
    parser.add_argument('--grade_scale', type=str, required=True, help='Bin lookup to use for inference')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to record')
    parser.add_argument('--instructor_reliability', type=float, default=16, help='Instructor reliability for calibrations')
    parser.add_argument('--seed', type=int, help='Random seed for Gibbs sampler')
    parser.add_argument('--verbose', action='store_true', help='Print status while running inference')
    # hyperparams
    parser.add_argument('--mu_s',      type=float, required=True)
    parser.add_argument('--sigma_s',   type=float, required=True)
    parser.add_argument('--alpha_tau', nargs='*', default= [], required=True)
    parser.add_argument('--beta_tau', nargs='*', default=[], required=True)
    parser.add_argument('--mu_b',   type=float, required=True)
    parser.add_argument('--sigma_b',    type=float, required=True)
    parser.add_argument('--p_uniform', '--p_uniform_high_effort', type=float, default=0.0), 
    parser.add_argument('--p_uniform_low_effort', type=float, default=0.0),  # unused. 

    # unused--- only to avoid errors
    parser.add_argument('--sigma_tau', type=float, default=0.0)
    parser.add_argument('--lambda_tau', type=float, default=0.0)
    parser.add_argument('--tau_l',     type=float, default=0.0)
    parser.add_argument('--alpha_e',   type=float, default=0.0)
    parser.add_argument('--beta_e',    type=float, default=0.0)

    args = parser.parse_args()
    verboseprint = print if args.verbose else lambda *a, **k: None
    # verboseprint(args)

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
    # Put together hyperparams
    if len(args.alpha_tau) > 1:
        hyperparams = (
            args.mu_s,
            args.sigma_s,
            args.mu_b,
            args.sigma_b,
            np.array(args.alpha_tau, dtype=float),
            np.array(args.beta_tau, dtype=float),
            args.p_uniform,
        )
    else: 
        hyperparams = (
            args.mu_s,
            args.sigma_s,
            args.mu_b,
            args.sigma_b,
            float(args.alpha_tau[0]),
            float(args.beta_tau[0]),
            args.p_uniform,
        )
        
    # Clamp efforts and reliabilities
    calibration_grades_no_dim = calibration_grades.sum(axis=1)
    reported_grades_instructor, graph_instructor = add_calibrations(
        reported_grades,
        graph,
        np.nan_to_num(calibration_grades),
        1 * ~np.isnan(calibration_grades_no_dim)
    )

    efforts_clamped = np.full((len(students))+1, np.nan)
    reliabilities_clamped = np.full((len(students))+1, np.nan)
    
    # Clamp TA efforts
    for i, (student, role) in enumerate(students.items()):
        if role == 'ta':
            efforts_clamped[i] = 1.0

    # Clamp instructor effort and reliability
    efforts_clamped[-1] = 1.0
    reliabilities_clamped[-1] = args.instructor_reliability

    # TODO: clamp zombies

    instructor_effort_grade = np.empty((1,effort_grades.shape[1]))
    instructor_effort_grade[:] = np.NaN
    effort_grades_instructor = np.concatenate([effort_grades,instructor_effort_grade],axis=0)
 
    clamped_values = {
        'efforts': efforts_clamped,
        'reliabilities': reliabilities_clamped,
        'responsibilities' : effort_grades_instructor
    }

    # Rescale grades
    if args.grade_scale == '5':
        reported_grades_instructor = convertGradeScale(reported_grades_instructor)

    # Run sampler
    verboseprint('Running inference...')
    if args.seed is not None:
        np.random.seed(args.seed)
    
    true_grades_gibbs, biases_gibbs, reliabilities_gibbs = censored_PG1_inference.run_gibbs(
        reported_grades_instructor,
        graph_instructor, 
        hyperparams, 
        max_error=None, # unused parameter... 
        clamped_values=clamped_values, 
        num_samples=args.num_samples,
        verbose=args.verbose,
    )

    # Save
    verboseprint(f'Saving samples to database...')
    sample_data = {
        'true_grades': true_grades_gibbs,
        'biases': biases_gibbs,
        'reliabilities': reliabilities_gibbs,
        'instructor_graph': graph_instructor,
    }    
    sample_args = vars(args).copy()
    del sample_args['database']

    with create_session(args.database, verbose=args.verbose) as session:
        save_samples(session, model_run, sample_data, sample_args, commit=True)