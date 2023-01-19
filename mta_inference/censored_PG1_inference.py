from tqdm import tqdm
import scipy
from scipy import stats
import numpy as np

from component_model_inference import get_clamp_masks, set_up_initial_point

# take core functions from other inference methods
from censored_model_inference import (
    norm_cdf, 
    sample_categorical, 
    convert_ll_to_prob,
    default_true_grade_samples,
    default_reliability_samples
)

from PG1_inference import (
    get_clamp_masks,
    set_up_initial_point
)

bin_edges = {
    0.0: (-100.0,  0.5),
    1.0: (  0.5,  1.5),
    2.0: (  1.5,  2.5),
    3.0: (  2.5,  3.5),
    4.0: (  3.5,  4.5),
    5.0: (  4.5,  100),
}


def get_lower_edge(reported_grade):

    return bin_edges[reported_grade][0]

def get_upper_edge(reported_grade):

    return bin_edges[reported_grade][1]

default_true_grade_grid = np.linspace(0, 7, 101)
default_reliability_grid = np.linspace(0, 10, 101)[1:] # don't include reliability of 0.0


# minimum bin probability to avoid -infinity log likelihoods
epsilon = 1e-6

# number of bins to sample uniformly over
# TODO: don't hard code
num_bins = 6

# grids/samples to use for bias updates
default_bias_grid = np.linspace(-3, 3, 61)
default_bias_samples = 100

def reported_grade_log_likelihoods(lower_edges, upper_edges, mask, true_grades, reliabilities, biases, p_uniform):
    """
    Find the log likelihood of every (bin, true grade, reliability, effort draw).

    Inputs:
    - lower_edges: matrix containing left edge of every reported grade's bin
    - upper_edges: matrix containing right edge of every reported grade's bin
    - mask: (graders, assignments, components, samples) matrix indicating which likelihoods to compute 
            (1 = compute; 0 = ignore) 
    - true_grades: matrix of true grades
    - reliabilities: matrix of reliabilities
    - biases: matrix of effort draws
    - mu_s: true grade mean hyperparameter

    Note that the mask input must be full dimension; all other inputs only need to broadcast to mask's shape

    Outputs:
    - matrix of likelihoods, same shape as mask. Equal to 0 wherever mask = 0. 
    """

    means = true_grades + biases
    precisions = reliabilities

    mask_idx = np.where(mask)    
    means_flat = np.broadcast_to(means, mask.shape)[mask_idx]
    precisions_flat = np.broadcast_to(precisions, mask.shape)[mask_idx]
    lower_edges_flat = np.broadcast_to(lower_edges, mask.shape)[mask_idx]
    upper_edges_flat = np.broadcast_to(upper_edges, mask.shape)[mask_idx]
    
    cdf_lbs = norm_cdf(lower_edges_flat, means_flat, precisions_flat)
    cdf_ubs = norm_cdf(upper_edges_flat, means_flat, precisions_flat)

    # save memory by deleting unneeded variables now
    del means_flat
    del precisions_flat
    del lower_edges_flat
    del upper_edges_flat

    bin_probabilities = p_uniform * (1/num_bins) + (1 - p_uniform) * (cdf_ubs - cdf_lbs) + epsilon

    lls = np.zeros(mask.shape) 
    lls[mask_idx] = np.log(bin_probabilities)
    return lls

def sample_true_grades(lower_edges, upper_edges, graph, reliabilities, biases, mu_s, sigma_s, p_uniform, use_grid=False):
    """
    Return a new sample of the true grade matrix.

    Inputs:
    - lower_edges: (graders, assignments, components) matrix with left edge of each reported grade bin
    - upper_edges: (graders, assignments, components) matrix with right edge of each reported grade bin
    - graph: (graders, assignments, 1) matrix indicating which grades are truly reported
    - reliabilities: (graders, 1, 1) matrix of reliabilities
    - biases: (graders, 1, 1) matrix of biases
    - mu_s: true grade mean hyperparameter
    - use_grid: if true, consider samples from a uniform grid; else, consider samples from prior

    Outputs:
    - (1, assignments, components) matrix of true grades
    """
    (num_graders, num_assignments, num_components) = lower_edges.shape
    num_samples = len(default_true_grade_grid) if use_grid else default_true_grade_samples
    full_shape = (num_graders, num_assignments, num_components, num_samples)
    graph_samples = np.broadcast_to(graph[:, :, :, np.newaxis], full_shape).astype(bool)

    if use_grid:
        true_grade_samples = np.tile(default_true_grade_grid, (1, num_assignments, num_components, 1))
        reported_lls = reported_grade_log_likelihoods(
            lower_edges[:, :, :, np.newaxis],
            upper_edges[:, :, :, np.newaxis],
            graph_samples,
            true_grade_samples,
            reliabilities[:, :, :, np.newaxis],
            biases[:, :, :, np.newaxis],
            p_uniform,
        ) * graph_samples
        prior_lls = np.tile(
            scipy.stats.norm.logpdf(default_true_grade_grid, mu_s, sigma_s), 
            (1, num_assignments, num_components, 1)
        )
        true_grade_lls = reported_lls.sum(axis=0, keepdims=True) + prior_lls

    else:
        true_grade_samples = stats.norm.rvs(
            mu_s, scale=sigma_s, size=(1, num_assignments, num_components, default_true_grade_samples)
        )
        reported_lls = reported_grade_log_likelihoods(
            lower_edges[:, :, :, np.newaxis],
            upper_edges[:, :, :, np.newaxis],
            graph_samples,
            true_grade_samples,
            reliabilities[:, :, :, np.newaxis],
            biases[:, :, :, np.newaxis],
            p_uniform,
        ) * graph_samples
        true_grade_lls = reported_lls.sum(axis=0, keepdims=True)

    true_grade_probabilities = convert_ll_to_prob(true_grade_lls)
    true_grades = sample_categorical(
        true_grade_samples, true_grade_probabilities
    )
    return true_grades

def sample_reliabilities(lower_edges, upper_edges, graph, true_grades, biases, alpha_rel, beta_rel, p_uniform, use_grid=False):
    """
    Return a new sample of the reliability matrix.

    Inputs:
    - lower_edges: (graders, assignments, components) matrix with left edge of each reported grade bin
    - upper_edges: (graders, assignments, components) matrix with right edge of each reported grade bin
    - graph: (graders, assignments, 1) matrix indicating which grades are truly reported
    - true_grades: (1, assignments, components) matrix of true grades
    - effort_draws: (graders, assignments, 1) matrix of effort draws
    - alpha_rel, beta_rel: reliability hyperparameters
    - use_grid: if true, consider samples from a uniform grid; else, consider samples from prior

    Outputs:
    - (graders, 1, 1) matrix of reliabilities

    TODO: previous implementation had:

        grade_errors = (observed_grades - true_grades)**2
        if max_error is not None:
           grade_errors = np.minimum(grade_errors, max_error**2)

    Is there an equivalent here to avoid wrecking a student's reliability with one very bad grade?
    Maybe mixing in some amount of uniform distribution with their true grade reported distribution?
    (This is effectively what epsilon does already!)
    """

    (num_graders, num_assignments, num_components) = lower_edges.shape
    num_samples = len(default_reliability_grid) if use_grid else default_reliability_samples
    full_shape = (num_graders, num_assignments, num_components, num_samples)

    graph_samples = np.broadcast_to(graph[:, :, :, np.newaxis], full_shape).astype(bool)
    
    if use_grid:
        reliability_samples = np.tile(default_reliability_grid, (num_graders, 1, 1, 1))
        reported_lls = reported_grade_log_likelihoods(
            lower_edges[:, :, :, np.newaxis],
            upper_edges[:, :, :, np.newaxis],
            graph_samples,
            true_grades[:, :, :, np.newaxis],
            reliability_samples,
            biases[:, :, :, np.newaxis],
            p_uniform,
        ) * graph_samples
        prior_lls = np.tile(
            scipy.stats.gamma.logpdf(default_reliability_grid, alpha_rel, scale=1/beta_rel),
            (num_graders, 1, 1, 1)
        )
        reliability_lls = reported_lls.sum(axis=(1, 2), keepdims=True) + prior_lls

    else:
        reliability_samples = stats.gamma.rvs(
            alpha_rel, scale=1/beta_rel, size=(num_graders, 1, 1, default_reliability_samples)
        )
        reported_lls = reported_grade_log_likelihoods(
            lower_edges[:, :, :, np.newaxis],
            upper_edges[:, :, :, np.newaxis],
            graph_samples,
            true_grades[:, :, :, np.newaxis],
            reliability_samples,
            biases[:, :, :, np.newaxis],
            p_uniform,
        ) * graph_samples
        reliability_lls = reported_lls.sum(axis=(1, 2), keepdims=True)

    reliability_probabilities = convert_ll_to_prob(reliability_lls)
    reliabilities = sample_categorical(
        reliability_samples, reliability_probabilities
    )
    return reliabilities

def sample_biases(lower_edges, upper_edges, graph, true_grades, reliabilities, mu_b, sigma_b, p_uniform, use_grid):
    """
    Return a new sample of the bias matrix.

    Inputs:
    - lower_edges: (graders, assignments, components) matrix with left edge of each reported grade bin
    - upper_edges: (graders, assignments, components) matrix with right edge of each reported grade bin
    - graph: (graders, assignments, 1) matrix indicating which grades are truly reported
    - true_grades: (1, assignments, components) matrix of true grades
    - reliabilities: (graders, 1, 1) matrix of reliabilities
    - mu_b, sigma_b: bias hyperparameter
    - tau_l: low effort precision hyperparameter

    Outputs:
    - (graders, 1, 1) matrix of reliabilities

    (note: uses "grid" approach rather than "sampling" approach because effort draw grid only has 2 values)
    """

    (num_graders, num_assignments, num_components) = lower_edges.shape
    num_samples = len(default_bias_grid) if use_grid else default_bias_samples
    full_shape = (num_graders, num_assignments, num_components, num_samples)

    graph_samples = np.broadcast_to(graph[:, :, :, np.newaxis], full_shape).astype(bool)

    if use_grid:
        bias_samples = np.tile(default_bias_grid, (num_graders, 1, 1, 1))
        reported_lls = reported_grade_log_likelihoods(
            lower_edges[:, :, :, np.newaxis],
            upper_edges[:, :, :, np.newaxis],
            graph_samples,
            true_grades[:, :, :, np.newaxis],
            reliabilities[:, :, :, np.newaxis],
            bias_samples,
            p_uniform,
        ) * graph_samples
        prior_lls = np.tile(
            scipy.stats.norm.logpdf(default_bias_grid, mu_b, scale=sigma_b),
            (num_graders, 1, 1, 1)
        )
        bias_lls = reported_lls.sum(axis=(1, 2), keepdims=True) + prior_lls

    else:
        bias_samples = stats.norm.rvs(
            mu_b, scale=sigma_b, size=(num_graders, 1, 1, default_bias_samples)
        )
        reported_lls = reported_grade_log_likelihoods(
            lower_edges[:, :, :, np.newaxis],
            upper_edges[:, :, :, np.newaxis],
            graph_samples,
            true_grades[:, :, :, np.newaxis],
            reliabilities[:, :, :, np.newaxis],
            bias_samples,
            p_uniform,
        ) * graph_samples
        bias_lls = reported_lls.sum(axis=(1, 2), keepdims=True)

    bias_probabilities = convert_ll_to_prob(bias_lls)
    biases = sample_categorical(
        bias_samples, bias_probabilities
    )
    return biases

def run_gibbs(
    reported_grades, graph, hyperparams, 
    initial_point=None, clamped_values={}, max_error=None, num_samples=1000,
    use_grid=True, verbose=True,
):
    """
    Inputs:
    - reported_grades: num_graders x num_assignments x num_components matrix of grades
    - graph: num_graders x num_assignments matrix: 1 if grader v graded assignment u; 0 otherwise
    - hyperparams: tuple of (mu_s, sigma_s, mu_b, sigma_b, alpha_rel, beta_rel)
    - initial_point: TODO: not implemented. leave as None.
    - clamped_values: dict of values to clamp. See get_clamp_masks() for format
    - max_error: maximum error (in points) used in reliability updates. No limit if None (default)
    - num_samples: number of Gibbs samples to take
    - use_grid: if True, use grid-based sampler; else, use prior-based sampler
    - verbose: if True, show progress bar
    """

    # Cast inputs to numpy arrays
    reported_grades = np.array(reported_grades, dtype=np.float)

    # Convert reported grades to bins
    reported_lower_edges = np.vectorize(get_lower_edge)(reported_grades)
    reported_upper_edges = np.vectorize(get_upper_edge)(reported_grades)
    
    # Unpack + process hyperparams
    (mu_s, sigma_s, mu_b, sigma_b, alpha_rel, beta_rel, p_uniform) = hyperparams

    # Set up initial point
    (num_graders, num_assignments, num_components) = reported_grades.shape
    true_grades, reliabilities, biases = set_up_initial_point(hyperparams, num_graders, num_assignments, num_components, initial_point)
    effort_draws = np.full((num_graders, num_assignments, 1), 1.0)
    graph = graph.reshape(num_graders, num_assignments, 1)

    # Get masks for clamped values
    (true_grades_clamped, reliabilities_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments, num_components    
    )

    # Set up return values: index by (sample number, grader/assignment number, component number [for true grades])
    ret_true_grades = np.zeros((num_samples, num_assignments, num_components))
    ret_reliabilities = np.zeros((num_samples, num_graders))
    ret_biases = np.zeros((num_samples, num_graders))

    for t in tqdm(range(num_samples), disable=not verbose):
        # Sample true grades
        true_grades = sample_true_grades(
            reported_lower_edges, reported_upper_edges, graph, reliabilities, biases, mu_s, sigma_s, p_uniform, use_grid
        )
        true_grades[~true_grades_clamped.mask] = true_grades_clamped[~true_grades_clamped.mask]

        # Sample biases for each grader
        biases = sample_biases(
            reported_lower_edges, reported_upper_edges, graph, true_grades, reliabilities, mu_b, sigma_b, p_uniform, use_grid
        )

        # Sample reliabilities
        reliabilities = sample_reliabilities(
            reported_lower_edges, reported_upper_edges, graph, true_grades, biases, alpha_rel, beta_rel, p_uniform, use_grid
        )
        reliabilities[~reliabilities_clamped.mask] = reliabilities_clamped[~reliabilities_clamped.mask]

        # Save samples
        ret_true_grades[t, :, :] = true_grades.reshape(num_assignments, num_components)
        ret_reliabilities[t, :] = reliabilities.flatten()
        ret_biases[t, :] = biases.flatten()

        
    # TODO: put these in a dataclass?
    return (ret_true_grades, ret_biases, ret_reliabilities)
