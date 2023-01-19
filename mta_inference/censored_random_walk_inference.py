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

from random_walk_inference import get_clamp_masks
from censored_model_inference import get_lower_edge, get_upper_edge
from inference_utils import sample_clipped_normal

# Avoid rounding issues with probabilities close to 0
epsilon = 1e-12

def sample_observed_grades(graph, reported_grades, true_grades, reliabilities, effort_draws, mu_s, tau_l):
    """
    Helper function: run sparse sampling for observed grades

    means_sparse =      [3.9, 4.6, 4.0  , ...]
    precisions_sparse = [2.0, 0.6, tau_l, ...]
    """
    
    (num_graders, num_assignments, num_components) = reported_grades.shape

    # Only update grades that were really observed
    graph_components = np.tile(graph, (1, 1, num_components))
    update_idx = np.where(graph_components == 1)

    reported_sparse = reported_grades[update_idx]
    # num_update = len(reported_sparse)

    high_effort = effort_draws == 1
    high_effort_sparse = high_effort[update_idx[0], update_idx[1], 0] # ignore component axis

    means_sparse = true_grades[0, update_idx[1], update_idx[2]] # ignore grader axis
    means_sparse[high_effort_sparse == 0] = mu_s

    precisions_sparse = reliabilities[update_idx[0], update_idx[1], 0] # ignore assignment and component axes
    precisions_sparse[high_effort_sparse == 0] = tau_l

    observed_lbs = np.vectorize(get_lower_edge)(reported_sparse)
    observed_ubs = np.vectorize(get_upper_edge)(reported_sparse)


    observed_sparse = sample_clipped_normal(
        means_sparse, precisions_sparse, None, observed_lbs, observed_ubs
    )
    # idx = np.argmax(observed_sparse)
    # print(means_sparse[idx], precisions_sparse[idx], observed_lbs[idx], observed_ubs[idx], observed_sparse[idx])
    # print(observed_sparse.max())

    # Fill in result
    observed_grades = np.zeros((num_graders, num_assignments, num_components))
    observed_grades[update_idx] = observed_sparse
    return observed_grades

def run_gibbs(
    reported_grades, ordered_graph, hyperparams, 
    initial_point=None, clamped_values={}, max_error=None, num_samples=1000,
    verbose=True,
):
    """
    Inputs:
    - reported_grades: num_graders x num_assignments x num_components matrix of grades
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
    reported_grades = np.array(reported_grades, dtype=np.float)

    # Unpack + process hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_tau, beta_tau, alpha_a, beta_a, tau_l) = hyperparams
    tau_s = 1 / sigma_s**2

    # Recover graph
    (num_graders, num_assignments, num_components) = reported_grades.shape
    graph = (ordered_graph >= 0)
    graph = graph.reshape(num_graders, num_assignments, 1)

    # Set up initial point
    if initial_point is None:
        true_grades = np.full((1, num_assignments, num_components), mu_s)
        efforts = np.full((num_graders, 1, 1), alpha_e / (alpha_e + beta_e))
        effort_draws = np.full((num_graders, num_assignments, 1), 1.0)
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
    ret_effort_draws = np.zeros((num_samples, num_graders, num_assignments))
    ret_true_grades = np.zeros((num_samples, num_assignments, num_components))
    ret_observed_grades = np.zeros((num_samples, num_graders, num_assignments, num_components))
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

        # Sample observed grades
        observed_grades = sample_observed_grades(graph, reported_grades, true_grades, reliabilities_expanded, effort_draws, mu_s, tau_l)

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

        ret_effort_draws[sample, :, :] = effort_draws.reshape(num_graders, num_assignments)  
        ret_true_grades[sample, :, :] = true_grades.reshape(num_assignments, num_components)
        ret_observed_grades[t, :, :, :] = observed_grades
        ret_efforts[sample, :] = efforts.flatten()
        for student in range(num_graders):
            ret_reliabilities[sample][student] = reliabilities[student]

    return (ret_effort_draws, ret_true_grades, ret_observed_grades, ret_efforts, ret_reliabilities)
