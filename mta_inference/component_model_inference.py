"""
Binary model, but splitting grades into four components.
"""

import scipy
from scipy import stats
import numpy as np
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, disable):
        for i, x in enumerate(iterable):
            if not disable:
                print(i)
            yield x

from mta_inference.inference_utils import sample_clipped_beta, sample_clipped_gamma
from mta_inference.component_data_generating_process import ComponentModelParams

# Avoid rounding issues with probabilities close to 0
epsilon = 1e-12

def get_clamp_masks(clamped_variables, num_graders, num_assignments, num_components):
    """
    Get masked arrays from dict of clamp values.

    Example: 
        clamped = {
            'true_grades': [[4, 4, 3, 3], [3, 5, 5, 4], np.nan, np.nan], 
        }
        masks = get_clamp_masks(clamped, 4, 4)

    Inputs:
    - clamped_variables: dict, with array of values to clamp for each variable (or np.nan for unclamped). 
        Valid keys are 'true_grades', 'efforts', 'reliabilities', 'responsibilities'.
        If key is missing, assume nothing clamped for this variable.
    - num_graders: used to make effort/reliability/responsibility lists, if missing
    - num_assignments: used to make true grade/responsibility lists, if missing
    - num_components: used to make true grades, if missing

    Outputs:
    - tuple of np.ma.arrays for (true grades, efforts, reliabilities, responsibilities)
    """

    # Initialize to default masks of correct shapes
    mask_shapes = {
        'true_grades': (1, num_assignments, num_components),
        'efforts': (num_graders, 1, 1),
        'reliabilities': (num_graders, 1, 1),
        'responsibilities': (num_graders, num_assignments, 1)
    }
    ret = {key: np.ma.masked_invalid(np.full(mask_shapes[key], np.nan, dtype=np.float)) for key in mask_shapes}

    # Replace with real values, if present
    for key in mask_shapes:
        if key not in clamped_variables or clamped_variables[key] is None:
            continue
        clamped_floats = np.array(clamped_variables[key], dtype=np.float)
        ret[key] = np.ma.masked_invalid(clamped_floats).reshape(mask_shapes[key])

    return (
        ret['true_grades'], ret['efforts'], ret['reliabilities'], ret['responsibilities']
    )

def set_up_initial_point(hyperparams, num_graders, num_assignments, num_components, initial_point, verbose=False):
    """
    Utility function: get initial point for EM and Gibbs and reshape to correct shapes
    """

    if initial_point is None:
        if verbose:
            print('No initial point provided; using prior means')
        # (mu_s, _, alpha_e, beta_e, alpha_rel, beta_rel, _) = hyperparams
        (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l, p_uniform) = hyperparams
        # True grades are row vector
        true_grades = np.full((1, num_assignments, num_components), mu_s)
        # Reliabilities and efforts are column vectors 
        reliabilities = np.full((num_graders, 1, 1), alpha_rel/beta_rel)
        efforts = np.full((num_graders, 1, 1), alpha_e/(alpha_e+beta_e))

    else:
        if verbose:
            print('Starting from provided initial point')
        (true_grades, efforts, reliabilities) = initial_point
        true_grades = true_grades.reshape(1, num_assignments, num_components)
        reliabilities = reliabilities.reshape(num_graders, 1, 1)
        efforts = efforts.reshape(num_graders, 1, 1)

    return true_grades, reliabilities, efforts

def get_constraints(C):
    """
    Get constraint functions for region
        reliability <= C*effort/(1-effort)
    or, equivalently,
        effort >= reliability / (C + reliability)

    Input:
    - C: tuning parameter for constraints

    Output:
    - tuple of (min effort function, max reliability function)
    """
    
    def effort_min(reliability):
        return reliability / (C + reliability)

    def reliability_max(effort):
        return C*effort / (1 - effort)

    return (effort_min, reliability_max)

def calculate_log_likelihood(observed_grades, graph, true_grades, reliabilities, effort_draws, component_model_params):
    # Unpack hyperparams
    mu_s = component_model_params.mu_s
    tau_l = component_model_params.tau_l

    # Reshape inputs
    (num_graders, num_assignments, num_components) = observed_grades.shape
    graph = graph.reshape(num_graders, num_assignments, 1)
    true_grades = true_grades.reshape(1, num_assignments, num_components)
    reliabilities = reliabilities.reshape(num_graders, 1, 1)
    effort_draws = effort_draws.reshape(num_graders, num_assignments, 1)

    # Only compute likelihood of grades that were really observed
    graph_components = np.tile(graph, (1, 1, num_components))
    grade_idx = np.where(graph_components == 1)
    grades_sparse = observed_grades[grade_idx]

    # Compute distribution params
    high_effort = effort_draws == 1
    high_effort_sparse = high_effort[grade_idx[0], grade_idx[1], 0] # ignore component axis

    means_sparse = true_grades[0, grade_idx[1], grade_idx[2]] # ignore grader axis
    means_sparse[high_effort_sparse == 0] = mu_s

    precisions_sparse = reliabilities[grade_idx[0], 0, 0] # ignore assignment and component axes
    precisions_sparse[high_effort_sparse == 0] = tau_l

    # Compute likeihoods
    log_likelihoods = stats.norm.logpdf(grades_sparse, loc=means_sparse, scale=1/np.sqrt(precisions_sparse))
    return np.sum(log_likelihoods)


def run_em(
    observed_grades, graph, hyperparams, 
    initial_point=None, clamped_values={}, max_error=None, num_iters=10, 
    constraints=None, verbose=True,
):
    """
    Inputs:
    - observed_grades: num_graders x num_assignments x num_components matrix of grades
    - graph: num_graders x num_assignments matrix: 1 if grader v graded assignment u; 0 otherwise
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l)
    - initial_point: arrays to use for first iteration; defaults to prior means
    - clamped_values: dict of values to clamp. See get_clamp_masks() for format
    - max_error: maximum error (in points) used in reliability updates. No limit if None (default)
    - num_iters: number of EM iterations to run
    - constraints: tuple of (min effort function, max reliability function)
        - min effort function: map reliability to minimum effort allowed
        - max reliability function: map effort to maximum effort allowed
    - verbose: if True, show progress bar
    
    TODO: add verbose param to save history?
    TODO: check for convergence?
    """

    # Cast inputs to numpy arrays
    observed_grades = np.array(observed_grades, dtype=np.float)

    # Unpack + process hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l, p_uniform) = hyperparams
    tau_s = 1 / sigma_s**2
    if constraints is not None:
        (effort_min, reliability_max) = constraints

    # Initialize parameter estimates
    (num_graders, num_assignments, num_components) = observed_grades.shape
    true_grades, reliabilities, efforts = set_up_initial_point(hyperparams, num_graders, num_assignments, num_components, initial_point)
    graph = graph.reshape(num_graders, num_assignments, 1)

    # Get masks for clamped values
    (true_grades_clamped, efforts_clamped, reliabilities_clamped, responsibilities_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments, num_components
    )

    for t in tqdm(range(num_iters), disable=not verbose):
        # Responsibilities
        pdf_high_effort = stats.norm.pdf(observed_grades, true_grades, 1 / np.sqrt(reliabilities))
        pdf_low_effort  = stats.norm.pdf(observed_grades, mu_s, 1 / np.sqrt(tau_l))
        pdf_high_effort = np.prod(pdf_high_effort, axis=2, keepdims=True) + epsilon
        pdf_low_effort  = np.prod(pdf_low_effort , axis=2, keepdims=True) + epsilon
        responsibilities = efforts * pdf_high_effort / (efforts * pdf_high_effort + (1 - efforts) * pdf_low_effort)
        # Set responsibilities to 0 unless grade actually observed
        responsibilities = responsibilities * graph
        # Restore clamped values
        responsibilities[~responsibilities_clamped.mask] = responsibilities_clamped[~responsibilities_clamped.mask]

        # True grades
        true_grades_prec = (responsibilities * reliabilities).sum(axis=0, keepdims=True) + tau_s 
        true_grades_num = (responsibilities * reliabilities * observed_grades).sum(axis=0, keepdims=True) + tau_s * mu_s 
        true_grades_mean = true_grades_num / true_grades_prec
        true_grades = true_grades_mean
        true_grades[~true_grades_clamped.mask] = true_grades_clamped[~true_grades_clamped.mask]

        # Efforts
        efforts_alpha = (graph * responsibilities).sum(axis=1, keepdims=True) + alpha_e
        efforts_beta = (graph * (1 - responsibilities)).sum(axis=1, keepdims=True) + beta_e 
        efforts = (efforts_alpha - 1) / (efforts_alpha + efforts_beta - 2)
        if constraints is not None:
            efforts = np.maximum(efforts, effort_min(reliabilities))
        efforts[~efforts_clamped.mask] = efforts_clamped[~efforts_clamped.mask]

        # Reliabilities
        grade_errors = (observed_grades - true_grades)**2
        if max_error is not None:
            grade_errors = np.minimum(grade_errors, max_error**2)
        total_grade_errors = grade_errors.sum(axis=2, keepdims=True)

        reliabilities_alpha = responsibilities.sum(axis=1, keepdims=True)*num_components/2 + alpha_rel
        reliabilities_beta = (responsibilities * total_grade_errors).sum(axis=1, keepdims=True)/2 + beta_rel
        reliabilities = np.maximum((reliabilities_alpha - 1), 0) / reliabilities_beta
        if constraints is not None:
            reliabilities = np.minimum(reliabilities, reliability_max(efforts))
        reliabilities[~reliabilities_clamped.mask] = reliabilities_clamped[~reliabilities_clamped.mask]

    return (
        responsibilities.reshape(num_graders, num_assignments),
        true_grades.reshape(num_assignments, num_components), 
        efforts.flatten(), 
        reliabilities.flatten()
    )

def run_gibbs(
    observed_grades, graph, hyperparams, 
    initial_point=None, clamped_values={}, max_error=None, num_samples=1000,
    use_grid=True, save_effort_draws=False, constraints=None, grade_scale='5', verbose=True, 
):
    """
    Inputs:
    - observed_grades: num_graders x num_assignments x num_components matrix of grades
    - graph: num_graders x num_assignments matrix: 1 if grader v graded assignment u; 0 otherwise
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l)
    - initial_point: tuple of (true grades array, effort array, reliability array), or None to use prior means
    - clamped_values: dict of values to clamp. See get_clamp_masks() for format
    - max_error: maximum error (in points) used in reliability updates. No limit if None (default)
    - num_samples: number of samples to take
    - constraints: tuple of (min effort function, max reliability function)
        - min effort function: map reliability to minimum effort allowed
        - max reliability function: map effort to maximum effort allowed
    - verbose: if True, show progress bar
    
    TODO: run with burn-in period? (can implement by taking slice of samples later...)
    """

    # Cast inputs to numpy arrays
    observed_grades = np.array(observed_grades, dtype=np.float)

    # Unpack + process hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l, p_uniform) = hyperparams
    component_model_params = ComponentModelParams(
        mu_s=mu_s,
        sigma_s=sigma_s,
        alpha_e=alpha_e, 
        beta_e=beta_e, 
        alpha_tau=alpha_rel, 
        beta_tau=beta_rel, 
        # NOTE: no TA hyperparams
        tau_l=tau_l,
    )
    tau_s = 1 / sigma_s**2
    if constraints is not None:
        (effort_min, reliability_max) = constraints

    # Set up initial point
    (num_graders, num_assignments, num_components) = observed_grades.shape
    true_grades, reliabilities, efforts = set_up_initial_point(hyperparams, num_graders, num_assignments, num_components, initial_point)
    graph = graph.reshape(num_graders, num_assignments, 1)

    # Get masks for clamped values
    (true_grades_clamped, efforts_clamped, reliabilities_clamped, effort_draws_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments, num_components    
    )

    # Set up return values: index by (sample number, grader/assignment number, component number [for true grades])
#    ret_effort_draws = np.zeros((num_samples, num_graders, num_assignments))
    ret_true_grades = np.zeros((num_samples, num_assignments, num_components))
    ret_efforts = np.zeros((num_samples, num_graders))
    ret_reliabilities = np.zeros((num_samples, num_graders))
    ret_likelihoods = np.zeros((num_samples, ))

    for t in tqdm(range(num_samples), disable=not verbose):
        # Sample effort draws for each assignment
        pdf_high_effort = stats.norm.pdf(observed_grades, true_grades, 1 / np.sqrt(reliabilities))
        pdf_low_effort  = stats.norm.pdf(observed_grades, mu_s, 1 / np.sqrt(tau_l))
        pdf_high_effort = np.prod(pdf_high_effort, axis=2, keepdims=True) + epsilon
        pdf_low_effort  = np.prod(pdf_low_effort , axis=2, keepdims=True) + epsilon
        responsibilities = efforts * pdf_high_effort / (efforts * pdf_high_effort + (1 - efforts) * pdf_low_effort)
        responsibilities = responsibilities * graph
        effort_draws = stats.bernoulli.rvs(responsibilities)
        effort_draws[~effort_draws_clamped.mask] = effort_draws_clamped[~effort_draws_clamped.mask]

        # Sample true grades
        true_grades_prec = (effort_draws * reliabilities).sum(axis=0, keepdims=True) + tau_s
        true_grades_num  = (effort_draws * reliabilities * observed_grades).sum(axis=0, keepdims=True) + tau_s * mu_s 
        true_grades_mean = true_grades_num / true_grades_prec
        true_grades = stats.norm.rvs(true_grades_mean, 1 / np.sqrt(true_grades_prec), size=(1, num_assignments, num_components))
        true_grades[~true_grades_clamped.mask] = true_grades_clamped[~true_grades_clamped.mask]

        # Sample efforts
        efforts_alpha = alpha_e + (graph * effort_draws).sum(axis=1, keepdims=True)
        efforts_beta = beta_e + (graph * (1 - effort_draws)).sum(axis=1, keepdims=True)
        min_efforts = None if constraints is None else effort_min(reliabilities) 
        efforts = sample_clipped_beta(efforts_alpha, efforts_beta, size=(num_graders, 1, 1), min_val=min_efforts)
        efforts[~efforts_clamped.mask] = efforts_clamped[~efforts_clamped.mask]

        # Sample reliabilities
        grade_errors = (observed_grades - true_grades)**2
        if max_error is not None:
            grade_errors = np.minimum(grade_errors, max_error**2)
        total_grade_errors = grade_errors.sum(axis=2, keepdims=True)
        reliabilities_alpha = effort_draws.sum(axis=1, keepdims=True)*num_components/2 + alpha_rel
        reliabilities_beta = (effort_draws * total_grade_errors).sum(axis=1, keepdims=True)/2 + beta_rel
        max_reliabilities = None if constraints is None else reliability_max(efforts)
        reliabilities = sample_clipped_gamma(reliabilities_alpha, reliabilities_beta, size=(num_graders, 1, 1), max_val=max_reliabilities)
        reliabilities[~reliabilities_clamped.mask] = reliabilities_clamped[~reliabilities_clamped.mask]

        # Save samples
#        ret_effort_draws[t, :, :] = effort_draws.reshape(num_graders, num_assignments)  
        ret_true_grades[t, :, :] = true_grades.reshape(num_assignments, num_components)
        ret_efforts[t, :] = efforts.flatten()
        ret_reliabilities[t, :] = reliabilities.flatten()

        # Calculate likelihood
        ret_likelihoods[t] = calculate_log_likelihood(
            observed_grades, graph, true_grades, reliabilities, effort_draws, component_model_params
        )

    return (ret_true_grades, ret_efforts, ret_reliabilities, ret_likelihoods) 
#    return (ret_effort_draws, ret_true_grades, ret_efforts, ret_reliabilities)

def run_gibbs_one_student(
    observed_grades, true_grades, hyperparams, 
    initial_point=None, max_error=None, num_samples=1000,
    constraints=None, verbose=True,
):
    """
    Run inference on a single student's effort and reliability.

    Inputs:
    - observed_grades: num_assignments x num_components array of observed grades from this student
    - true_grades: num_assignments x num_components array of true grades for each of these assignments 
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l)
    - initial_point: tuple of (effort, reliability), or None to start at prior mean
    - max_error: maximum error (in points) used in reliability updates. No limit if None (default)
    - num_samples: number of Gibbs samples to take
    - constraints: tuple of (min effort function, max reliability function)
        - min effort function: map reliability to minimum effort allowed
        - max reliability function: map effort to maximum effort allowed
    - verbose: if True, show progress bar

    Outputs: (list of effort samples, list of reliability samples)
    """

    # Cast inputs to numpy arrays
    observed_grades = np.array(observed_grades, dtype=np.float)
    true_grades = np.array(true_grades, dtype=np.float)

    # Unpack hyperparams
    # (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l) = hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l, p_uniform) = hyperparams
    num_assignments, num_components = observed_grades.shape
    if constraints is not None:
        (effort_min, reliability_max) = constraints

    # Set up initial point
    if initial_point is None:
        effort = alpha_e / (alpha_e + beta_e)
        reliability = alpha_rel / beta_rel
    else:
        (effort, reliability) = initial_point

    # Set up return values: index by sample number
    ret_efforts = np.zeros((num_samples,))
    ret_reliabilities = np.zeros((num_samples,))

    for t in tqdm(range(num_samples), disable=not verbose):
        # Sample effort draws for each assignment
        pdf_high_effort = stats.norm.pdf(observed_grades, true_grades, 1 / np.sqrt(reliability))
        pdf_low_effort  = stats.norm.pdf(observed_grades, mu_s, 1 / np.sqrt(tau_l))
        pdf_high_effort = np.prod(pdf_high_effort, axis=1) + epsilon
        pdf_low_effort  = np.prod(pdf_low_effort , axis=1) + epsilon
        responsibilities = effort * pdf_high_effort / (effort * pdf_high_effort + (1 - effort) * pdf_low_effort)
        # Cast effort draws to numpy array in case there's only one value
        effort_draws = stats.bernoulli.rvs(responsibilities, size=num_assignments)    

        # Sample effort
        effort_alpha = alpha_e + effort_draws.sum()
        effort_beta = beta_e + (1 - effort_draws).sum()
        min_effort = None if constraints is None else effort_min(reliability) 
        effort = sample_clipped_beta(effort_alpha, effort_beta, min_val=min_effort)
        ret_efforts[t] = effort

        # Sample reliability
        grade_errors = (observed_grades - true_grades)**2
        if max_error is not None:
            grade_errors = np.minimum(grade_errors, max_error**2)
        total_grade_errors = grade_errors.sum(axis=1)
        reliability_alpha = alpha_rel + effort_draws.sum()*2
        reliability_beta = beta_rel + (effort_draws * total_grade_errors).sum()/2
        max_reliability = None if constraints is None else reliability_max(effort)
        reliability = sample_clipped_gamma(reliability_alpha, reliability_beta, max_val=max_reliability)
        ret_reliabilities[t] = reliability

    return (ret_efforts, ret_reliabilities)

def get_dependability_one_student(
    observed_grades, true_grades, hyperparams, quantiles, 
    initial_point=None, max_error=None, num_samples=1000,
    constraints=None, verbose=True,
):
    """
    Summarize Gibbs samples for a single student.

    Inputs:
    - observed_grades: num_assignments x num_components array of observed grades from this student
    - true_grades: num_assignments x num_components array of true grades for each of these assignments 
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l)
    - quantiles: floats or list of floats of quantiles to compute (ex: [5, 30, 95] for 5th, 30th, and 95th percentiles)
    - initial_point: tuple of (effort, reliability), or None to start at prior mean
    - max_error: maximum error (in points) used in reliability updates. No limit if None (default)
    - num_samples: number of Gibbs samples to take
    - constraints: tuple of (min effort function, max reliability function)
        - min effort function: map reliability to minimum effort allowed
        - max reliability function: map effort to maximum effort allowed
    - verbose: if True, show progress bar

    Outputs: (mean dependability, list of dependability quantiles) for student
    """

    effort_samples, reliability_samples = run_gibbs_one_student(
        observed_grades, true_grades, hyperparams, 
        initial_point=initial_point, max_error=max_error, num_samples=num_samples,
        constraints=constraints, verbose=verbose
    )
    dependability_samples = effort_samples * reliability_samples

    dependability_mean = np.mean(dependability_samples)
    dependability_quantiles = np.percentile(dependability_samples, quantiles)
    return (dependability_mean, dependability_quantiles)

def calculate_marginal_likelihood(observed_grades, graph, component_model_params, grader_roles, num_samples=10000):  
    (num_graders, num_assignments, num_components) = observed_grades.shape
    (grader_idx, assignment_idx) = np.where(graph)

    # pre-compute distributions
    mu_s = component_model_params.mu_s
    sigma_s = component_model_params.sigma_s
    true_grade_dist = scipy.stats.norm(mu_s, sigma_s)
    
    alpha_e = component_model_params.alpha_e
    beta_e = component_model_params.beta_e

    alpha_tau = component_model_params.alpha_tau
    alpha_tau_ta = component_model_params.alpha_tau_ta
    beta_tau = component_model_params.beta_tau
    beta_tau_ta = component_model_params.beta_tau_ta
    reliability_alpha = np.array([alpha_tau_ta if role == 'ta' else alpha_tau for role in grader_roles])
    reliability_beta = np.array([beta_tau_ta if role == 'ta' else beta_tau for role in grader_roles])

    likelihoods = []
    for sample in tqdm(range(num_samples)):
        # TODO: refactor with component model generator
        # Generate true grades
        true_grades = true_grade_dist.rvs(size=(num_assignments, num_components))

        # Generate efforts (but modify TA efforts to 1.0)
        efforts = scipy.stats.beta.rvs(alpha_e, beta_e, size=num_graders)
        efforts[np.array(grader_roles) == 'ta'] = 1.0 # TODO: clamp TA efforts at 1?

        efforts_sparse = efforts[grader_idx]
        effort_draws = np.zeros((num_graders, num_assignments))
        effort_draws[grader_idx, assignment_idx] = scipy.stats.bernoulli.rvs(efforts_sparse)
        
        # Generate reliabilities (using different distributions for students and TAs)
        reliabilities = scipy.stats.gamma.rvs(reliability_alpha, scale=1/reliability_beta)

        likelihoods.append(calculate_log_likelihood(observed_grades, graph, true_grades, reliabilities, effort_draws, component_model_params))

    return np.array(likelihoods)
