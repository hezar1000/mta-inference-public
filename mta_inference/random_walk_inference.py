"""
Component model, but reliabilities can change over time
"""

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
        Valid keys are 'true_grades', 'efforts', 'responsibilities'.
        If key is missing, assume nothing clamped for this variable.
    - num_graders: used to make effort/responsibility lists, if missing
    - num_assignments: used to make true grade/responsibility lists, if missing
    - num_components: used to make true grades, if missing

    Outputs:
    - tuple of np.ma.arrays for (true grades, efforts, responsibilities)
    """

    # Initialize to default masks of correct shapes
    mask_shapes = {
        'true_grades': (1, num_assignments, num_components),
        'efforts': (num_graders, 1, 1),
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
        ret['true_grades'], ret['efforts'], ret['responsibilities']
    )

def run_gibbs(
    observed_grades, ordered_graph, hyperparams, 
    initial_point=None, clamped_values={}, max_error=None, num_samples=1000,
    verbose=True,
):
    """
    Inputs:
    - observed_grades: num_graders x num_assignments x num_components matrix of grades
    - ordered_graph: num_graders x num_assignments matrix: 
        - if grader v graded assignment u, timestep that this grade occured at; 
        - otherwise, -1
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_tau, beta_tau, alpha_a, beta_a, tau_l)
    - initial_point: tuple of (true grades array, effort array, reliability array), or None to use prior means
    - clamped_values: dict of values to clamp. See get_clamp_masks() for format
    - max_error: maximum error (in points) used in reliability updates. No limit if None (default)
    - num_samples: number of samples to take
    - verbose: if True, show progress bar
    """

    # Cast inputs to numpy arrays
    observed_grades = np.array(observed_grades, dtype=np.float)

    # Unpack + process hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_tau, beta_tau, alpha_a, beta_a, tau_l) = hyperparams
    tau_s = 1 / sigma_s**2

    # Recover graph
    (num_graders, num_assignments, num_components) = observed_grades.shape
    graph = (ordered_graph >= 0)
    graph = graph.reshape(num_graders, num_assignments, 1)

    # Set up initial point
    if initial_point is None:
        true_grades = np.full((1, num_assignments, num_components), mu_s)
        efforts = np.full((num_graders, 1, 1), alpha_e / (alpha_e + beta_e))
        reliabilities = []
        auxiliary = []
        num_assignments_per_student = np.max(ordered_graph, axis=1) + 1
        num_assignments_per_student[num_assignments_per_student == 0] = 1

        for student in range(num_graders):
            reliabilities_student = np.zeros(num_assignments_per_student[student])
            auxiliary_student = np.zeros(num_assignments_per_student[student])

            auxiliary_student[0] = 1
            for i in range(num_assignments_per_student[student]):
                if i > 0:
                    auxiliary_student[i] = alpha_a / (beta_a * reliabilities_student[i-1])
                reliabilities_student[i] = alpha_tau / (beta_tau * auxiliary_student[i])
            reliabilities.append(reliabilities_student)
            auxiliary.append(auxiliary_student)
    else:
        # TODO: add auxiliary variables to initialization
        raise NotImplementedError('Only supports default initial point for now.')


    # Get masks for clamped values
    # Note that reliabilities can't be clamped
    (true_grades_clamped, efforts_clamped, effort_draws_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments, num_components    
    )

    # Set up return values: index in order (sample number, grader, time, assignment, component)
#    ret_effort_draws = np.zeros((num_samples, num_graders, num_assignments))
    ret_true_grades = np.zeros((num_samples, num_assignments, num_components))
    ret_efforts = np.zeros((num_samples, num_graders))
    ret_reliabilities = [[None for _ in range(num_graders)] for _ in range(num_samples)]

    for sample in tqdm(range(num_samples), disable=not verbose):
        # Expand reliabilities
        reliabilities_expanded = np.zeros((num_graders, num_assignments, 1))
        for student in range(num_graders):
            reliability_indices = ordered_graph[student, :]
            
            # TODO: vectorize?
            for (assignment, reliability_index) in enumerate(reliability_indices):
                if reliability_index >= 0:
                    reliabilities_expanded[student, assignment, 0] = reliabilities[student][reliability_index]

        # Effort draws
        pdf_high_effort = stats.norm.pdf(observed_grades, true_grades, 1 / np.sqrt(reliabilities_expanded))
        pdf_high_effort = np.prod(pdf_high_effort, axis=2, keepdims=True) + epsilon
        pdf_low_effort  = stats.norm.pdf(observed_grades, mu_s, 1 / np.sqrt(tau_l))
        pdf_low_effort  = np.prod(pdf_low_effort , axis=2, keepdims=True) + epsilon
        effort_draw_probabilities = efforts * pdf_high_effort / (efforts * pdf_high_effort + (1 - efforts) * pdf_low_effort)
        effort_draw_probabilities = effort_draw_probabilities * graph
        effort_draws = stats.bernoulli.rvs(effort_draw_probabilities)
        effort_draws[~effort_draws_clamped.mask] = effort_draws_clamped[~effort_draws_clamped.mask]
    
        # True grades
        true_grades_prec = (effort_draws * reliabilities_expanded).sum(axis=0, keepdims=True) + tau_s
        true_grades_num  = (effort_draws * reliabilities_expanded * observed_grades).sum(axis=0, keepdims=True) + tau_s * mu_s 
        true_grades_mean = true_grades_num / true_grades_prec
        true_grades = stats.norm.rvs(true_grades_mean, 1 / np.sqrt(true_grades_prec), size=(1, num_assignments, num_components))
        true_grades[~true_grades_clamped.mask] = true_grades_clamped[~true_grades_clamped.mask]

        # Efforts
        efforts_alpha = alpha_e + (graph * effort_draws).sum(axis=1, keepdims=True)
        efforts_beta = beta_e + (graph * (1 - effort_draws)).sum(axis=1, keepdims=True)
        efforts = stats.beta.rvs(efforts_alpha, efforts_beta, size=(num_graders, 1, 1))
        efforts[~efforts_clamped.mask] = efforts_clamped[~efforts_clamped.mask]

        # Reliabilities (and auxiliary reliabilities) 
        for student in range(num_graders):
            # Reliabilities
            # First, condition on surrounding auxiliary variables
            reliabilities_alpha       = np.full(len(reliabilities[student]), alpha_tau)
            reliabilities_alpha[:-1] += alpha_a
            reliabilities_beta        = beta_tau * auxiliary[student]
            reliabilities_beta[:-1]  += beta_a * auxiliary[student][1:]

            ordered_graph_student = ordered_graph[student, :]
            for t in range(len(reliabilities[student])):
                assignments_t = np.where(ordered_graph_student == t)[0]
                grade_errors = (observed_grades[student, assignments_t, :] - true_grades[0, assignments_t, :])**2
                if max_error is not None:
                    grade_errors = np.minimum(grade_errors, max_error**2)
                total_grade_errors = grade_errors.sum(axis=1)

                reliabilities_alpha[t] += effort_draws[student, assignments_t, 0].sum()*2
                reliabilities_beta[t] += (effort_draws[student, assignments_t, 0] * total_grade_errors).sum()/2

            reliabilities[student] = stats.gamma.rvs(reliabilities_alpha, scale=1/reliabilities_beta, size=reliabilities[student].shape)
            # TODO: add ability to clamp reliabilities?

            # Auxiliary reliabilities (except auxiliary[0])
            auxiliary_alpha = alpha_a + alpha_tau
            auxiliary_betas = beta_a * reliabilities[student][:-1] + beta_tau * reliabilities[student][1:]
            auxiliary[student][1:] = stats.gamma.rvs(auxiliary_alpha, scale=1/auxiliary_betas, size=auxiliary[student][1:].shape)

#        ret_effort_draws[sample, :, :] = effort_draws.reshape(num_graders, num_assignments)  
        ret_true_grades[sample, :, :] = true_grades.reshape(num_assignments, num_components)
        ret_efforts[sample, :] = efforts.flatten()
        for student in range(num_graders):
            ret_reliabilities[sample][student] = reliabilities[student]

    return (ret_true_grades, ret_efforts, ret_reliabilities)
#    return (ret_effort_draws, ret_true_grades, ret_efforts, ret_reliabilities)


def run_gibbs_one_student(
    observed_grades, true_grades, hyperparams, 
    initial_point=None, max_error=None, num_samples=1000,
    verbose=True,
):
    """
    Run inference on a single student's effort and reliability.

    Inputs:
    - observed_grades: num_assignments x num_components array of observed grades from this student; assumed to be in chronological order
    - true_grades: num_assignments x num_components array of true grades for each of these assignments
    - hyperparams: tuple of (mu_s, sigma_s, alpha_e, beta_e, alpha_tau, beta_tau, alpha_a, beta_a, tau_l)
    - initial_point: tuple of (effort, reliability), or None to start at prior mean
    - max_error: maximum error (in points) used in reliability updates. No limit if None (default)
    - num_samples: number of Gibbs samples to take
    - verbose: if True, show progress bar

    Outputs: (list of effort samples, list of reliability samples)
    """

    # Cast inputs to numpy arrays
    observed_grades = np.array(observed_grades, dtype=np.float)
    true_grades = np.array(true_grades, dtype=np.float)

    # Unpack hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_tau, beta_tau, alpha_a, beta_a, tau_l) = hyperparams
    num_assignments, num_components = observed_grades.shape

    # Set up initial point
    if initial_point is None:
        effort = alpha_e / (alpha_e + beta_e)
        reliabilities = np.zeros(num_assignments)
        auxiliary = np.zeros(num_assignments)

        auxiliary[0] = 1
        for i in range(num_assignments):
            if i > 0:
                auxiliary[i] = alpha_a / (beta_a * reliabilities[i-1])
            reliabilities[i] = alpha_tau / (beta_tau * auxiliary[i])
    else:
        # TODO: add auxiliary variables to initialization
        raise NotImplementedError('Only supports default initial point for now.')
        # (effort, reliabilities) = initial_point

    # Set up return values: index by sample number
    ret_efforts = np.zeros((num_samples,))
    ret_reliabilities = np.zeros((num_samples, num_assignments))

    for t in tqdm(range(num_samples), disable=not verbose):
        # Sample effort draws for each assignment
        pdf_high_effort = stats.norm.pdf(observed_grades, true_grades, 1 / np.sqrt(reliabilities.reshape(-1, 1)))
        pdf_low_effort  = stats.norm.pdf(observed_grades, mu_s, 1 / np.sqrt(tau_l))
        pdf_high_effort = np.prod(pdf_high_effort, axis=1) + epsilon
        pdf_low_effort  = np.prod(pdf_low_effort , axis=1) + epsilon
        responsibilities = effort * pdf_high_effort / (effort * pdf_high_effort + (1 - effort) * pdf_low_effort)
        # Cast effort draws to numpy array in case there's only one value
        effort_draws = stats.bernoulli.rvs(responsibilities, size=num_assignments)    

        # Sample effort
        effort_alpha = alpha_e + effort_draws.sum()
        effort_beta = beta_e + (1 - effort_draws).sum()
        effort = stats.beta.rvs(effort_alpha, effort_beta)
        ret_efforts[t] = effort

        # Sample auxiliary reliabilities (except auxiliary[0])
        auxiliary_alpha = alpha_a + alpha_tau
        auxiliary_betas = beta_a * reliabilities[:-1] + beta_tau * reliabilities[1:]
        auxiliary[1:] = stats.gamma.rvs(auxiliary_alpha, scale=1/auxiliary_betas)

        # Sample reliability
        grade_errors = (observed_grades - true_grades)**2
        if max_error is not None:
            grade_errors = np.minimum(grade_errors, max_error**2)
        total_grade_errors = grade_errors.sum(axis=1)
        reliability_alphas = alpha_tau + effort_draws*2
        reliability_alphas[:-1] += alpha_a
        reliability_betas = beta_tau * auxiliary + (effort_draws * total_grade_errors)/2
        reliability_betas[:-1] += beta_a * auxiliary[1:]
        reliabilities = stats.gamma.rvs(reliability_alphas, scale=1/reliability_betas)
        ret_reliabilities[t] = reliabilities

    return (ret_efforts, ret_reliabilities)
