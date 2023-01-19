"""
Implements all variants of PG5:
- censored/uncensored
- biases/no biases
- efforts/no efforts
"""

import gc
import numpy as np
from tqdm import tqdm
from scipy import stats

from .inference_utils import convert_to_bins, generate_bins, set_up_initial_point_PG5, get_clamp_masks, clamp_parameters, norm_cdf, sample_categorical, convert_ll_to_prob

# Minimum bin probability
EPSILON = 1e-12

# Mapping from grade scale to bin definitions
grade_bin_lookup = {
    '5': generate_bins([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], -20.0, 20.0),
    '25': generate_bins([0.0, 6.25, 12.5, 16.25, 20.0, 25.0], -100.0, 100.0)
}

# grids to use for grid-based parameter updates
# note: no grids necessary for efforts (conjugate priors) or effort draws (binary variable)
true_grade_grids = {
    '5': np.linspace(0, 6, 101),
    '25': np.linspace(0, 30, 101)
}

reliability_grids = {
    '5': np.linspace(0, 10, 101)[1:],
    '25': np.linspace(0, 2, 101)[1:],
}

bias_grids = {
    '5': np.linspace(-2, 2, 101),
    '25': np.linspace(-10, 10, 101)
}

def add_sample_axis(data):
    """
    Add fourth "sample" axis to each variable in data.
    """
    return {k: data[k][:, :, :, np.newaxis] for k in data}

def reported_grade_log_likelihoods(mask, expanded_data, expanded_sample, hyperparams):
    """
    Find the log likelihood of every (bin, true grade, reliability, bias, effort draw).

    Inputs:
    - mask: (graders, assignments, components, samples) matrix indicating which likelihoods to compute 
            (1 = compute; 0 = ignore) 
    - expanded_data: dict of 4D arrays of bins/bin edges and graph
    - expanded_sample: dict of 4D arrays of parameter values being tested
    - hyperparams: dict of hyperparams

    Note that the mask input must be full dimension; all other inputs only need to broadcast to mask's shape

    Outputs:
    - matrix of likelihoods, same shape as mask. Equal to 0 wherever mask = 0. 
    """

    # Flatten variables
    mask_idx = np.where(mask)    
    true_grades_flat = np.broadcast_to(expanded_sample['true_grades'], mask.shape)[mask_idx]
    reliabilities_flat = np.broadcast_to(expanded_sample['reliabilities'], mask.shape)[mask_idx]
    biases_flat = np.broadcast_to(expanded_sample['biases'], mask.shape)[mask_idx]
    effort_draws_flat = np.broadcast_to(expanded_sample['effort_draws'], mask.shape)[mask_idx]

    # Compute normal distribution parameters
    means_flat = effort_draws_flat * (true_grades_flat + biases_flat) + (1 - effort_draws_flat) * hyperparams['mu_s']
    if hyperparams['disable_correlation'] == True:
        precisions_flat = effort_draws_flat * reliabilities_flat + (1 - effort_draws_flat) * hyperparams['tau_l']
    else:
        precisions_flat = effort_draws_flat * reliabilities_flat / hyperparams['lambda_tau'] + (1 - effort_draws_flat) * hyperparams['tau_l']
    p_uniform_flat = effort_draws_flat * hyperparams['p_uniform_high_effort'] + (1 - effort_draws_flat) * hyperparams['p_uniform_low_effort']

    # Compute bin probabilities
    if hyperparams['disable_censoring'] == True:
        reported_grades_flat = np.broadcast_to(expanded_data['reported_grades'], mask.shape)[mask_idx]
        bin_probabilities = stats.norm.pdf(reported_grades_flat, means_flat, precisions_flat)
    else: # hyperparams['disable_censoring'] == False
        lower_edges_flat = np.broadcast_to(expanded_data['reported_lower_edges'], mask.shape)[mask_idx]
        upper_edges_flat = np.broadcast_to(expanded_data['reported_upper_edges'], mask.shape)[mask_idx]
        bin_probabilities = norm_cdf(upper_edges_flat, means_flat, precisions_flat) - norm_cdf(lower_edges_flat, means_flat, precisions_flat)

    # Mix with uniform distribution
    bin_probabilities = p_uniform_flat * (1/hyperparams['num_bins']) + (1 - p_uniform_flat) * bin_probabilities + EPSILON
    
    # Remove unneeded variables before final likelihood calculation
    if hyperparams['disable_censoring'] == False:
        del lower_edges_flat
        del upper_edges_flat       

    del effort_draws_flat
    del true_grades_flat
    del reliabilities_flat
    del means_flat
    del precisions_flat
    del p_uniform_flat
    gc.collect()

    # TODO: this array has size (graders, assignments, components, samples), and computing it is the main memory bottleneck.
    # This is a shame because it's very sparse (mostly 0s), and we immediately sum over one or more of the axes.
    # Find some way to avoid computing the entire matrix?
    lls = np.zeros(mask.shape) 
    lls[mask_idx] = np.log(bin_probabilities)

    return lls

def true_grade_log_likelihoods(author_mask, expanded_sample, hyperparams):
    """
    PG5: calculate the likelihood of each true grade conditioned on the author's reliability.
    """

    # Build parameters
    author_mask_idx = np.where(author_mask)    
    true_grades_flat = np.broadcast_to(expanded_sample['true_grades'], author_mask.shape)[author_mask_idx]
    reliabilities_flat = np.broadcast_to(expanded_sample['reliabilities'], author_mask.shape)[author_mask_idx]

    true_grade_precision = 1 / hyperparams['sigma_s']**2
    reliability_precision = 1 / hyperparams['sigma_tau']**2
    precisions_flat = true_grade_precision + reliability_precision
    means_flat = (hyperparams['mu_s'] * true_grade_precision + reliabilities_flat * reliability_precision) / precisions_flat

    # Calculate likelihoods
    log_likelihoods_flat = stats.norm.logpdf(true_grades_flat, means_flat, 1 / np.sqrt(precisions_flat))

    # Convert back 
    true_grade_lls = np.zeros(author_mask.shape)
    true_grade_lls[author_mask_idx] = log_likelihoods_flat
    return true_grade_lls


def sample_true_grades(data, current_sample, clamped_parameters, hyperparams, true_grade_grid):
    (num_graders, num_assignments, num_components, _) = data['reported_lower_edges'].shape
    num_samples = len(true_grade_grid)
    mask = np.broadcast_to(data['graph'], (num_graders, num_assignments, num_components, num_samples))
    author_mask = np.broadcast_to(data['author_graph'], (num_graders, num_assignments, num_components, num_samples))

    # build samples
    expanded_sample = add_sample_axis(current_sample)
    expanded_sample['true_grades'] = np.tile(true_grade_grid, (1, num_assignments, num_components, 1))

    # calculate priors
    if hyperparams['disable_correlation']:
        # PG1 prior
        true_grade_lls = stats.norm.logpdf(expanded_sample['true_grades'], hyperparams['mu_s'], hyperparams['sigma_s'])
    else:
        # PG5 prior
        # TODO: this returns the uniform distribution for grades with no author (calibrations)!
        # might not be a big deal because these often have lots of grades...
        true_grade_lls = true_grade_log_likelihoods(author_mask, expanded_sample, hyperparams).sum(axis=0, keepdims=True)

    # add reported grade likelihoods
    rg_lls = reported_grade_log_likelihoods(mask, data, expanded_sample, hyperparams).sum(axis=0, keepdims=True)
    true_grade_lls += rg_lls

    # sample from posteriors and clamp
    true_grade_probabilities = convert_ll_to_prob(true_grade_lls)
    current_sample['true_grades'] = sample_categorical(expanded_sample['true_grades'], true_grade_probabilities)
    current_sample = clamp_parameters(current_sample, clamped_parameters, 'true_grades')
    return current_sample

def sample_reliabilities(data, current_sample, clamped_parameters, hyperparams, reliability_grid):
    (num_graders, num_assignments, num_components, _) = data['reported_lower_edges'].shape
    num_samples = len(reliability_grid)
    mask = np.broadcast_to(data['graph'], (num_graders, num_assignments, num_components, num_samples))
    author_mask = np.broadcast_to(data['author_graph'], (num_graders, num_assignments, num_components, num_samples))

    # build samples
    expanded_sample = add_sample_axis(current_sample)
    expanded_sample['reliabilities'] = np.tile(reliability_grid, (num_graders, 1, 1, 1))

    # calculate reliability prior
    if hyperparams['disable_correlation']:
        # PG0/1 prior
        reliability_lls = stats.gamma.logpdf(
            expanded_sample['reliabilities'], 
            hyperparams['alpha_tau'],
            scale=1.0/hyperparams['beta_tau'],
        )
    else:
        # PG5 prior, conditioned on true grades
        reliability_lls = stats.norm.logpdf(
            expanded_sample['reliabilities'], 
            hyperparams['mu_s'], 
            np.sqrt(hyperparams['sigma_tau']**2 + hyperparams['sigma_s']**2)
        )

    # PG5 only: calculate true grade likelihoods
    if not hyperparams['disable_correlation']: 
        reliability_lls += true_grade_log_likelihoods(author_mask, expanded_sample, hyperparams).sum(axis=(1,2), keepdims=True)

    # calculate reported grade likelihoods
    reliability_lls += (reported_grade_log_likelihoods(mask, data, expanded_sample, hyperparams) * mask).sum(axis=(1,2), keepdims=True)
    
    # sample from posteriors and clamp
    reliability_probabilities = convert_ll_to_prob(reliability_lls)
    current_sample['reliabilities'] = sample_categorical(expanded_sample['reliabilities'], reliability_probabilities)
    current_sample = clamp_parameters(current_sample, clamped_parameters, 'reliabilities')
    return current_sample

def sample_biases(data, current_sample, clamped_parameters, hyperparams, bias_grid):
    (num_graders, num_assignments, num_components, _) = data['reported_lower_edges'].shape
    num_samples = len(bias_grid)
    mask = np.broadcast_to(data['graph'], (num_graders, num_assignments, num_components, num_samples))

    # build samples
    expanded_sample = add_sample_axis(current_sample)
    expanded_sample['biases'] = np.tile(bias_grid, (num_graders, 1, 1, 1))

    # calculate priors
    bias_lls = np.tile(
        stats.norm.logpdf(bias_grid, hyperparams['mu_b'], hyperparams['sigma_b']), 
        (num_graders, 1, 1, 1)
    )

    # add reported grade likelihoods
    bias_lls += (reported_grade_log_likelihoods(mask, data, expanded_sample, hyperparams) * mask).sum(axis=(1,2), keepdims=True)
    
    # sample from posteriors and clamp
    bias_probabilities = convert_ll_to_prob(bias_lls)
    current_sample['biases'] = sample_categorical(expanded_sample['biases'], bias_probabilities)
    current_sample = clamp_parameters(current_sample, clamped_parameters, 'biases')
    return current_sample

def sample_effort_draws(data, current_sample, clamped_parameters, hyperparams):
    (num_graders, num_assignments, num_components, _) = data['reported_lower_edges'].shape
    mask = np.broadcast_to(data['graph'], (num_graders, num_assignments, num_components, 2))

    # build samples
    expanded_sample = add_sample_axis(current_sample)
    expanded_sample['effort_draws'] = np.stack([
        np.full((num_graders, num_assignments, 1), 0),
        np.full((num_graders, num_assignments, 1), 1),
    ], axis=3)

    # calculate priors
    with np.errstate(divide='ignore'): # don't worry about log(0) issues
        prior_ll = np.log(np.stack([
            (1 - expanded_sample['efforts'])[:, :, :, 0],
            expanded_sample['efforts'][:, :, :, 0],
        ], axis=3))
    
    # calculate likelihoods
    reported_lls = reported_grade_log_likelihoods(mask, data, expanded_sample, hyperparams) * mask
    
    # sample from posteriors and clamp
    effort_draw_lls = reported_lls.sum(axis=2, keepdims=True) + prior_ll
    effort_draw_probabilities = convert_ll_to_prob(effort_draw_lls)
    effort_draw_probabilities_positive = effort_draw_probabilities[:, :, :, 1] * data['graph'][:, :, :, 0]
    current_sample['effort_draws'] = stats.bernoulli.rvs(effort_draw_probabilities_positive, size=(num_graders, num_assignments, 1))
    current_sample = clamp_parameters(current_sample, clamped_parameters, 'effort_draws')
    return current_sample

def sample_efforts(data, current_sample, clamped_parameters, hyperparams):
    # Get variables
    alpha_e = hyperparams['alpha_e']
    beta_e = hyperparams['beta_e']
    graph = data['graph']
    effort_draws = current_sample['effort_draws']
    num_graders = graph.shape[0]

    # Calculate closed-form posteriors and draw sample
    efforts_alpha = alpha_e + (graph[:, :, :, 0] * effort_draws).sum(axis=1, keepdims=True)
    efforts_beta = beta_e + (graph[:, :, :, 0] * (1 - effort_draws)).sum(axis=1, keepdims=True)
    current_sample['efforts'] = stats.beta.rvs(efforts_alpha, efforts_beta, size=(num_graders, 1, 1))

    # Clamp
    current_sample = clamp_parameters(current_sample, clamped_parameters, 'efforts')
    return current_sample

def run_gibbs(
    reported_grades, graph, author_graph, hyperparams, initial_point=None, clamped_values={},
    num_samples=1000, save_effort_draws=False, grade_scale='5', verbose=True, 
):
    """
    Inputs:
    - reported_grades: num_graders x num_assignments x num_components matrix of grades
    - graph: num_graders x num_assignments matrix: 1 if grader v graded assignment u; 0 otherwise
    - author_graph: num_graders x num_assignments matrix: 1 if grader v wrote assignment u; 0 otherwise
    - hyperparams: dict of hyperparams
    - clamped_values: dict of values to clamp. See get_clamp_masks() for format
    - num_samples: number of Gibbs samples to take
    - save_effort_draws: if True, record samples of effort draws 
    - grade_scale: selects bin edges and grids for true grades/reliabilities
    - verbose: if True, show progress bar
    """

    # Set up reported grades and graph
    reported_grades = np.array(reported_grades, dtype=np.float)
    (num_graders, num_assignments, num_components) = reported_grades.shape
    reported_lower_edges, reported_upper_edges = convert_to_bins(reported_grades, grade_bin_lookup[grade_scale])

    graph = graph.reshape(num_graders, num_assignments, 1)
    author_graph = author_graph.reshape(num_graders, num_assignments, 1)
    grader_idx, assignment_idx, component_idx = np.where(graph == 1)
    
    # Add sample axis to data
    data = add_sample_axis({
        'reported_grades': reported_grades,
        'reported_lower_edges': reported_lower_edges,
        'reported_upper_edges': reported_upper_edges,
        'graph': graph.astype(bool),
        'author_graph': author_graph.astype(bool),
    })

    # Process hyperparams
    hyperparams['tau_s'] = 1 / hyperparams['sigma_s']**2
    hyperparams['num_bins'] = int(grade_scale) + 1

    # Get masks for clamped values
    clamped_parameters = get_clamp_masks(clamped_values, num_graders, num_assignments, num_components)

    # Set up initial point
    current_sample = set_up_initial_point_PG5(hyperparams, num_graders, num_assignments, num_components, initial_point)
    current_sample = clamp_parameters(current_sample, clamped_parameters)

    # Get grids 
    true_grade_grid = true_grade_grids[grade_scale]
    reliability_grid = reliability_grids[grade_scale]
    bias_grid = bias_grids[grade_scale]

    # Set up return values: index by 
    samples = {
        'true_grades': np.zeros((num_samples, num_assignments, num_components)),
        'reliabilities': np.zeros((num_samples, num_graders)),
        'biases': np.zeros((num_samples, num_graders)),
        'efforts': np.zeros((num_samples, num_graders)),
    }

    # For memory tracking purposes: allocate the entire sample matrix now
    samples['true_grades'][:] = 0.0
    samples['reliabilities'][:] = 0.0
    samples['biases'][:] = 0.0
    samples['efforts'][:] = 0.0

    if save_effort_draws:
        samples['effort_draws'] = np.zeros((num_samples, int(graph.sum())), bool)
        samples['effort_draws'][:] = 0.0

    for t in tqdm(range(num_samples), disable=not verbose):
        # Sample each variable
        if not hyperparams['disable_efforts']:
            current_sample = sample_effort_draws(data, current_sample, clamped_parameters, hyperparams)
            current_sample = sample_efforts     (data, current_sample, clamped_parameters, hyperparams)
        
        if not hyperparams['disable_biases']:
            current_sample = sample_biases   (data, current_sample, clamped_parameters, hyperparams, bias_grid)
        
        current_sample = sample_true_grades  (data, current_sample, clamped_parameters, hyperparams, true_grade_grid)
        current_sample = sample_reliabilities(data, current_sample, clamped_parameters, hyperparams, reliability_grid)

        # Save samples
        for var_name in ['true_grades', 'reliabilities', 'biases', 'efforts']:
            save_shape = samples[var_name][t].shape
            samples[var_name][t, :] = current_sample[var_name].reshape(save_shape)
        
        # Save effort draws separately (+ compress sparse matrix)
        if save_effort_draws:
            samples['effort_draws'][t, :] = current_sample['effort_draws'][grader_idx, assignment_idx, 0]

    return samples
