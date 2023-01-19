import json
from tqdm import tqdm
from scipy import stats
import numpy as np

from inference_utils import get_ci

# Avoid rounding issues with probabilities close to 0
epsilon = 1e-6

def get_clamp_masks(clamped_variables, num_graders, num_assignments):
    """
    Get masked arrays from dict of clamp values.

    Example: 
        clamped = {
            'true_grades': [14, 15, np.nan, np.nan], 
        }
        masks = get_clamp_masks(clamped, 4, 4)

    Inputs:
    - clamped_variables: dict, with array of values to clamp for each variable (or np.nan for unclamped). 
        Valid keys are 'true_grades', 'efforts', 'reliabilities', 'responsibilities'.
        If key is missing, assume nothing clamped for this variable.
    - num_graders: used to make effort/reliability/responsibility lists, if missing
    - num_assignments: used to make true grade/responsibility lists, if missing

    Outputs:
    - tuple of np.ma.arrays for (true grades, efforts, reliabilities, responsibilities)
    """

    # Initialize to default masks of correct shapes
    mask_shapes = {
        'true_grades': (1, num_assignments),
        'efforts': (num_graders, 1),
        'reliabilities': (num_graders, 1),
        'responsibilities': (num_graders, num_assignments)
    }
    ret = {key: np.ma.masked_invalid(np.full(mask_shapes[key], np.nan)) for key in mask_shapes}

    # Replace with real values, if present
    for key in mask_shapes:
        if key not in clamped_variables or clamped_variables[key] is None:
            continue
        ret[key] = np.ma.masked_invalid(clamped_variables[key]).reshape(mask_shapes[key])

    return (
        ret['true_grades'], ret['efforts'], ret['reliabilities'], ret['responsibilities']
    )

def get_ci(samples, ci):
    """
    Get a confidence interval from samples.
    Example for 95% confidence interval:
        samples = np.random.random(100)
        ci_min, ci_max = get_ci(samples, 0.95)

    Inputs:
    - samples: list or numpy array of data
    - ci: fraction of mass to include in interval. 
          Resulting interval ranges from the (1-ci)/2th to (1+ci)/2th percentile

    Outputs:
    - Tuple of (ci_min, ci_max)
    """
    
    # Note: percentiles are numbers in [0, 100]
    lower_percentile = 100 * (1 - ci) / 2
    upper_percentile = 100 * (1 + ci) / 2
    [ci_min, ci_max] = np.percentile(samples, [lower_percentile, upper_percentile])
    return (ci_min, ci_max)

def run_em(observed_grades, graph, hyperparams, initial_point=None, clamped_values={}, num_iters=10):
    """
    Inputs:
    - observed_grades: num_graders x num_assignments matrix of grades
    - graph: num_graders x num_assignments matrix: 1 if grader v graded assignment u; 0 otherwise
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l)
    - num_iters: number of EM iterations to run
    
    TODO: add verbose param to save history?
    TODO: check for convergence?
    """

    # Unpack + process hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l) = hyperparams
    tau_s = 1 / sigma_s**2

    # Initialize parameter estimates
    (num_graders, num_assignments) = observed_grades.shape
    if initial_point is None:
        print('EM: No initial point provided; using prior means')
        # True grades are row vector
        true_grades = np.array([[mu_s]*num_assignments])
        # Reliabilities and efforts are column vectors 
        reliabilities = np.array([[alpha_rel/beta_rel]]*num_graders)
        # efforts = np.array([[alpha_e/(alpha_e+beta_e)]]*num_graders)
        efforts = np.array([[0.5]]*num_graders)
    else:
        print('EM: starting from provided initial point')
        (true_grades, efforts, reliabilities) = initial_point
        true_grades = true_grades.reshape(1, num_assignments)
        reliabilities = reliabilities.reshape(num_graders, 1)
        efforts = efforts.reshape(num_graders, 1)

    # Get masks for clamped values
    (true_grades_clamped, efforts_clamped, reliabilities_clamped, responsibilities_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments    
    )

    for t in tqdm(range(num_iters)):
        # Responsibilities
        pdf_high_effort = stats.norm.pdf(observed_grades, true_grades, 1 / np.sqrt(reliabilities)) + epsilon
        pdf_low_effort  = stats.norm.pdf(observed_grades, mu_s, 1 / np.sqrt(tau_l)) + epsilon
        responsibilities = efforts * pdf_high_effort / (efforts * pdf_high_effort + (1 - efforts) * pdf_low_effort)
        # Set responsibilities to 0 unless grade actually observed
        responsibilities = responsibilities * graph
        # Restore clamped values
        responsibilities[~responsibilities_clamped.mask] = responsibilities_clamped[~responsibilities_clamped.mask]

        # True grades
        # TODO: need to multiply responsibilities * graph here?
        true_grades_prec = (responsibilities * reliabilities).sum(axis=0, keepdims=True) + tau_s 
        true_grades_num = (responsibilities * reliabilities * observed_grades).sum(axis=0, keepdims=True) + tau_s * mu_s 
        true_grades_mean = true_grades_num / true_grades_prec
        true_grades = true_grades_mean
        true_grades[~true_grades_clamped.mask] = true_grades_clamped[~true_grades_clamped.mask]

        # Efforts
        efforts_alpha = (graph * responsibilities).sum(axis=1, keepdims=True) + alpha_e
        efforts_beta = (graph * (1 - responsibilities)).sum(axis=1, keepdims=True) + beta_e 
        efforts = (efforts_alpha - 1) / (efforts_alpha + efforts_beta - 2)
        efforts[~efforts_clamped.mask] = efforts_clamped[~efforts_clamped.mask]

        # Reliabilities
        reliabilities_alpha = responsibilities.sum(axis=1, keepdims=True)/2 + alpha_rel
        reliabilities_beta = (responsibilities * (observed_grades - true_grades)**2).sum(axis=1, keepdims=True)/2 + beta_rel
        reliabilities = np.max((reliabilities_alpha - 1), 0) / reliabilities_beta
        reliabilities[~reliabilities_clamped.mask] = reliabilities_clamped[~reliabilities_clamped.mask]

    return true_grades.flatten(), efforts.flatten(), reliabilities.flatten()

def run_gibbs(observed_grades, graph, hyperparams, initial_point=None, clamped_values={}, num_samples=1000):
    """
    Inputs:
    - observed_grades: num_graders x num_assignments matrix of grades
    - graph: num_graders x num_assignments matrix: 1 if grader v graded assignment u; 0 otherwise
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l)
    - initial_point: tuple of (true grades array, effort array, reliability array), or None for prior means
    - num_samples: number of samples to take
    
    TODO: run with burn-in period?
    """

    # Unpack + process hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l) = hyperparams
    tau_s = 1 / sigma_s**2

    # Set up initial point
    (num_graders, num_assignments) = observed_grades.shape
    if initial_point is None:
        print('No initial point provided; using prior means')

        # True grades are row vector
        true_grades = np.array([[mu_s]*num_assignments])
        # Reliabilities and efforts are column vectors 
        reliabilities = np.array([[alpha_rel/beta_rel]]*num_graders)
        efforts = np.array([[alpha_e/(alpha_e+beta_e)]]*num_graders)
    else:
        print('Starting Gibbs sampling from provided initial point')
        (true_grades, efforts, reliabilities) = initial_point
        true_grades = true_grades.reshape(1, num_assignments)
        reliabilities = reliabilities.reshape(num_graders, 1)
        efforts = efforts.reshape(num_graders, 1)

    # Get masks for clamped values
    (true_grades_clamped, efforts_clamped, reliabilities_clamped, effort_draws_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments    
    )

    # Set up return values: index by (sample number, grader/assignment number)
    ret_true_grades = np.zeros((num_samples, num_assignments))
    ret_efforts = np.zeros((num_samples, num_graders))
    ret_reliabilities = np.zeros((num_samples, num_graders))

    for t in tqdm(range(num_samples)):
        # Sample effort draws for each assignment
        pdf_high_effort = stats.norm.pdf(observed_grades, true_grades, 1 / np.sqrt(reliabilities)) + epsilon
        pdf_low_effort  = stats.norm.pdf(observed_grades, mu_s, 1 / np.sqrt(tau_l)) + epsilon
        responsibilities = efforts * pdf_high_effort / (efforts * pdf_high_effort + (1 - efforts) * pdf_low_effort)
        responsibilities = responsibilities * graph
        effort_draws = stats.bernoulli.rvs(responsibilities)
        effort_draws[~effort_draws_clamped.mask] = effort_draws_clamped[~effort_draws_clamped.mask]

        # Sample true grades
        true_grades_prec = (effort_draws * reliabilities).sum(axis=0, keepdims=True) + tau_s
        true_grades_num  = (effort_draws * reliabilities * observed_grades).sum(axis=0, keepdims=True) + tau_s * mu_s 
        true_grades_mean = true_grades_num / true_grades_prec
        true_grades = stats.norm.rvs(true_grades_mean, 1 / np.sqrt(true_grades_prec)).reshape(1, num_assignments)
        true_grades[~true_grades_clamped.mask] = true_grades_clamped[~true_grades_clamped.mask]

        # Sample efforts
        efforts_alpha = alpha_e + (graph * effort_draws).sum(axis=1, keepdims=True)
        efforts_beta = beta_e + (graph * (1 - effort_draws)).sum(axis=1, keepdims=True)
        efforts = stats.beta.rvs(efforts_alpha, efforts_beta).reshape(num_graders, 1)
        efforts[~efforts_clamped.mask] = efforts_clamped[~efforts_clamped.mask]

        # Sample reliabilities
        reliabilities_alpha = effort_draws.sum(axis=1, keepdims=True)/2 + alpha_rel
        reliabilities_beta = (effort_draws * (observed_grades - true_grades)**2).sum(axis=1, keepdims=True)/2 + beta_rel
        reliabilities = stats.gamma.rvs(reliabilities_alpha, scale = 1 / reliabilities_beta).reshape(num_graders, 1)
        reliabilities[~reliabilities_clamped.mask] = reliabilities_clamped[~reliabilities_clamped.mask]

        # Save samples
        ret_true_grades[t, :] = true_grades.flatten()
        ret_efforts[t, :] = efforts.flatten()
        ret_reliabilities[t, :] = reliabilities.flatten()

    return (ret_true_grades, ret_efforts, ret_reliabilities)

def run_gibbs_one_student(observed_grades, true_grades, hyperparams, initial_point=None, num_samples=1000):
    """
    Run inference on a single student's effort and reliability.

    Inputs:
    - observed_grades: list of observed grades from this student
    - true_grades: list of true grades for each of these assignments 
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l)
    - initial_point: tuple of (effort, reliability), or None to start at prior mean
    - num_samples: number of Gibbs samples to take

    Outputs: (list of effort samples, list of reliability samples)
    """

    # Cast inputs to numpy arrays
    observed_grades = np.array(observed_grades)
    true_grades = np.array(true_grades)

    # Unpack hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l) = hyperparams

    # Set up initial point
    if initial_point is None:
        effort = alpha_e / (alpha_e + beta_e)
        reliability = alpha_rel / beta_rel
    else:
        (effort, reliability) = initial_point

    # Set up return values: index by sample number
    ret_efforts = np.zeros((num_samples,))
    ret_reliabilities = np.zeros((num_samples,))

    for t in tqdm(range(num_samples)):
        # Sample effort draws for each assignment
        pdf_high_effort = stats.norm.pdf(observed_grades, true_grades, 1 / np.sqrt(reliability)) + epsilon
        pdf_low_effort  = stats.norm.pdf(observed_grades, mu_s, 1 / np.sqrt(tau_l)) + epsilon
        responsibilities = effort * pdf_high_effort / (effort * pdf_high_effort + (1 - effort) * pdf_low_effort)
        # Cast effort draws to numpy array in case there's only one value
        effort_draws = np.array(stats.bernoulli.rvs(responsibilities))    

        # Sample effort
        effort_alpha = alpha_e + effort_draws.sum()
        effort_beta = beta_e + (1 - effort_draws).sum()
        effort = stats.beta.rvs(effort_alpha, effort_beta)
        ret_efforts[t] = effort

        # Sample reliability
        reliability_alpha = alpha_rel + effort_draws.sum()/2
        reliability_beta = beta_rel + (effort_draws * (observed_grades - true_grades)**2).sum()/2
        reliability = stats.gamma.rvs(reliability_alpha, scale = 1 / reliability_beta)
        ret_reliabilities[t] = reliability

    return (ret_efforts, ret_reliabilities)

def get_dependability_cis_one_student(observed_grades, true_grades, hyperparams, ci, initial_point=None, num_samples=1000):
    """
    Summarize Gibbs samples into dependability confidence interval for a single student.

    Inputs:
    - observed_grades: list of observed grades from this student
    - true_grades: list of true grades for each of these assignments 
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l)
    - ci: float in (0.0, 1.0): width of confidence interval (ex: 0.95 for 95% confidence interval)
    - initial_point: tuple of (effort, reliability), or None to start at prior mean
    - num_samples: number of Gibbs samples to take

    Outputs: (min dependability, mean dependability, max dependability) for student
    """

    effort_samples, reliability_samples = run_gibbs_one_student(observed_grades, true_grades, hyperparams, initial_point, num_samples)
    dependability_samples = np.multiply(effort_samples, reliability_samples)
    dependability_mean = np.mean(dependability_samples)
    (dependability_min, dependability_max) = get_ci(dependability_samples, ci)
    return (dependability_min, dependability_mean, dependability_max)
