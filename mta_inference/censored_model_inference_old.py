from tqdm import tqdm
from scipy import stats
import numpy as np

from component_model_inference import get_clamp_masks, set_up_initial_point
from inference_utils import sample_clipped_normal, truncnorm_stats

epsilon = 1e-6

bin_edges = {
    0.0: (0.0, 0.5),
    1.0: (0.5, 1.5),
    2.0: (1.5, 2.5),
    3.0: (2.5, 3.5),
    4.0: (3.5, 4.5),
    5.0: (4.5, 5.0),
}

def get_lower_edge(reported_grade):
    return bin_edges[reported_grade][0]

def get_upper_edge(reported_grade):
    return bin_edges[reported_grade][1]


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

    precisions_sparse = reliabilities[update_idx[0], 0, 0] # ignore assignment and component axes
    precisions_sparse[high_effort_sparse == 0] = tau_l

    observed_lbs = np.vectorize(get_lower_edge)(reported_sparse)
    observed_ubs = np.vectorize(get_upper_edge)(reported_sparse)


    observed_sparse = sample_clipped_normal(
        means_sparse, precisions_sparse, None, observed_lbs, observed_ubs
    )
    print(observed_sparse.argmax())
    print(observed_sparse[660])
    print(means_sparse[660])
    print(precisions_sparse[660])
    # idx = np.argmax(observed_sparse)
    # print(means_sparse[idx], precisions_sparse[idx], observed_lbs[idx], observed_ubs[idx], observed_sparse[idx])
    # print(observed_sparse.max())

    # Fill in result
    observed_grades = np.zeros((num_graders, num_assignments, num_components))
    observed_grades[update_idx] = observed_sparse
    return observed_grades

def run_gibbs(
    reported_grades, graph, hyperparams, 
    initial_point=None, clamped_values={}, max_error=None, num_samples=1000,
    constraints=None, verbose=True,
):
    """
    Inputs:
    - reported_grades: num_graders x num_assignments x num_components matrix of grades
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
    """

    # Cast inputs to numpy arrays
    reported_grades = np.array(reported_grades, dtype=np.float)

    # Unpack + process hyperparams
    (mu_s, sigma_s, alpha_e, beta_e, alpha_rel, beta_rel, tau_l) = hyperparams
    tau_s = 1 / sigma_s**2
    if constraints is not None:
        (effort_min, reliability_max) = constraints

    # Set up initial point
    (num_graders, num_assignments, num_components) = reported_grades.shape
    true_grades, reliabilities, efforts = set_up_initial_point(hyperparams, num_graders, num_assignments, num_components, initial_point)
    # observed_grades = reported_grades.copy()
    effort_draws = np.full((num_graders, num_assignments, 1), 1.0)
    graph = graph.reshape(num_graders, num_assignments, 1)

    # Get masks for clamped values
    (true_grades_clamped, efforts_clamped, reliabilities_clamped, effort_draws_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments, num_components    
    )

    # Set up return values: index by (sample number, grader/assignment number, component number [for true grades])
#    ret_effort_draws = np.zeros((num_samples, num_graders, num_assignments))
    ret_true_grades = np.zeros((num_samples, num_assignments, num_components))
    ret_observed_grades = np.zeros((num_samples, num_graders, num_assignments, num_components))
    ret_efforts = np.zeros((num_samples, num_graders))
    ret_reliabilities = np.zeros((num_samples, num_graders))

    for t in tqdm(range(num_samples), disable=not verbose):
        # Sample observed grades
        observed_grades = sample_observed_grades(graph, reported_grades, true_grades, reliabilities, effort_draws, mu_s, tau_l)

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
        efforts = stats.beta.rvs(efforts_alpha, efforts_beta, size=(num_graders, 1, 1))
        efforts[~efforts_clamped.mask] = efforts_clamped[~efforts_clamped.mask]

        # Sample reliabilities
        grade_errors = (observed_grades - true_grades)**2
        if max_error is not None:
            grade_errors = np.minimum(grade_errors, max_error**2)
        total_grade_errors = grade_errors.sum(axis=2, keepdims=True)
        reliabilities_alpha = effort_draws.sum(axis=1, keepdims=True)*2 + alpha_rel
        reliabilities_beta = (effort_draws * total_grade_errors).sum(axis=1, keepdims=True)/2 + beta_rel
        reliabilities = stats.gamma.rvs(reliabilities_alpha, scale=1/reliabilities_beta)
        reliabilities[~reliabilities_clamped.mask] = reliabilities_clamped[~reliabilities_clamped.mask]

        # Save samples
#        ret_effort_draws[t, :, :] = effort_draws.reshape(num_graders, num_assignments)  
        ret_true_grades[t, :, :] = true_grades.reshape(num_assignments, num_components)
        ret_observed_grades[t, :, :, :] = observed_grades
        ret_efforts[t, :] = efforts.flatten()
        ret_reliabilities[t, :] = reliabilities.flatten()

        # Compute likelihood
        
        

    return (ret_true_grades, ret_observed_grades, ret_efforts, ret_reliabilities)
#    return (ret_effort_draws, ret_true_grades, ret_observed_grades, ret_efforts, ret_reliabilities)
