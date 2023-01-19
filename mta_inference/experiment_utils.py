"""
Helpful functions for running/evaluating inference.
"""

from collections import OrderedDict
import numpy as np
import scipy.stats
import scipy.special

from mta_inference.censored_model_inference import convert_to_bins

def splitZombieGraph(graph, ordered_graph, reported_grades, effort_grades, students, window_size):
    """
    Split the class into separate graphs for zombie inference and regular inference.
    """

    num_reviews_per_student = np.sum(graph,axis=1)

    # Create variables for zombie inference...
    zombie_graph = np.copy(graph)
    zombie_reported_grades = np.copy(reported_grades)
    zombie_effort_grades = np.copy(effort_grades)

    # ...and variables for real inference
    real_graph = np.copy(graph)
    real_reported_grades = np.copy(reported_grades)
    real_effort_grades = np.copy(effort_grades)
    real_graph_zombies = np.copy(graph)
    real_reported_grades_zombies = np.copy(reported_grades)
    real_effort_grades_zombies = np.copy(effort_grades)

    # Zero out things that real/zombie students don't grade 
    for i,num_reviews in enumerate(num_reviews_per_student):
        # case 1: more than 2x window reviews
        if int(num_reviews) >= 2 * window_size:
            # Real students only grade final window
            for j in range(int(num_reviews)-window_size):
                index = np.where(ordered_graph[i]==j)
                real_graph[i,index]=0
                real_effort_grades[i,index]=0
                real_reported_grades[i,index,:] = np.array([0,0,0,0])
                
            # Zombies don't grade final window
            for k in range(int(num_reviews)-window_size,int(num_reviews)):
                index = np.where(ordered_graph[i]==k)
                zombie_graph[i,index]=0
                real_graph_zombies[i,index]=0
                zombie_effort_grades[i,index]=0
                real_effort_grades_zombies[i,index]=0
                zombie_reported_grades[i,index,:] = np.array([0,0,0,0])
                real_reported_grades_zombies[i,index,:] = np.array([0,0,0,0])

        # case 2: more than 1 window, but less than 2 windows
        elif int(num_reviews) <  2 * window_size and int(num_reviews) > window_size:
            # Real students only grade final window
            for j in range(int(num_reviews)-window_size):
                index = np.where(ordered_graph[i]==j)
                real_graph[i,index]=0
                real_effort_grades[i,index]=0
                real_reported_grades[i,index,:] = np.array([0,0,0,0])
                
            # Zombies grade first window (including overlap)
            for k in range(window_size,int(num_reviews)):
                index = np.where(ordered_graph[i]==k)
                zombie_graph[i,index]=0
                zombie_effort_grades[i,index]=0
                zombie_reported_grades[i,index,:] = np.array([0,0,0,0])

            # In final inference, zombies don't grade final window
            for k in range(int(num_reviews)-window_size,int(num_reviews)):
                index = np.where(ordered_graph[i]==k)
                real_graph_zombies[i,index]=0
                real_effort_grades_zombies[i,index]=0
                real_reported_grades_zombies[i,index,:] = np.array([0,0,0,0])
        else:
            # case 3: less than 1 window
            # real students grade everything; zombies grade nothing
            for j in range(int(num_reviews)):
                index = np.where(ordered_graph[i]==j)
                zombie_graph[i,index]=0
                real_graph_zombies[i,index]=0
                zombie_effort_grades[i,index]=0
                real_effort_grades_zombies[i,index]=0
                zombie_reported_grades[i,index,:] = np.array([0,0,0,0])
                real_reported_grades_zombies[i,index,:] = np.array([0,0,0,0])
                
    # Remove all zombie TA grades
    for i, (student, role) in enumerate(students.items()):
        if role == 'ta':
            real_graph_zombies[i,:]= 0
            real_graph[i,:]= graph[i,:]
            zombie_graph[i,:]=0
            real_reported_grades[i,:,:] = reported_grades[i,:,:]
            real_effort_grades[i,:] = effort_grades[i,:]
            zombie_effort_grades[i,:]=0
            real_effort_grades_zombies[i,:]=0
            zombie_reported_grades[i,:,:] = 0
            real_reported_grades_zombies[i,:,:] = 0

    # Build full inference graph by combining real students and zombies
    final_graph = np.concatenate((real_graph, real_graph_zombies), axis=0)
    final_effort_grades = np.concatenate((real_effort_grades, real_effort_grades_zombies), axis=0)
    final_reported_grades = np.concatenate((real_reported_grades, real_reported_grades_zombies), axis=0)

    # Create zombie student list and add to final list
    zombie_students = OrderedDict()
    for key in students:
        zombie_students[key+'_zumbie'] = students[key]
    final_students = OrderedDict(list(students.items()) + list(zombie_students.items()))

    return (
        zombie_graph,
        zombie_reported_grades,
        zombie_effort_grades, 
        zombie_students,
        final_graph,
        final_effort_grades, 
        final_reported_grades,
        final_students,
    )

def calculateLikelihoods(
    graph, 
    reported_grades, 
    true_grades, 
    reliabilities, 
    uniform_prob_high_effort=0.0,
    uniform_prob_low_effort=0.0,
    num_bins=6,
    effort_draws=None, 
    efforts=None, 
    mu_s=None, 
    tau_l=None, 
    biases=None, 
    reduce_per_assignment=True,
):
    """
    Compute the log likelihood of each grade for a set of BEM/PG1 parameters.

    Inputs:
    - graph, reported_grades, true_grades, reliabilities: as in inference
    - effort_draws: as in inference. if None, assume that P(effort_draw) = effort.
    - efforts: as in inference. if None, assume all 1.
    - biases: as in inference. if None, assume all 0.
    - uniform_prob_high_effort, uniform_prob_low_effort (float): probability of playing uniform random strategy
    - num_bins (int): number of possible grades
    - mu_s, tau_l: hyperparams. only necessary if efforts are not None.
    - reduce_per_assignment: if True, sum log likelihoods across reports for each student grade

    Outputs: np array of log likelihoods (per assignment if reduce_per_assignment=True; per reported grade otherwise)
    """
    (num_graders, num_assignments, num_components) = reported_grades.shape
    
    # set default values: 100% high effort, 0 bias
    if effort_draws is None and efforts is None:
        mu_s = 4  # unused, arbitrary value when all students high effort
        tau_l = 1 # unused, arbitrary value when all students high effort
    if efforts is None:
        efforts = np.full((num_graders, 1, 1), 1.0)
    if effort_draws is None:
        effort_draws = np.tile(efforts, (1, num_assignments, 1))
    if biases is None:
        biases = np.full((num_graders, 1, 1), 0.0)

    # make shapes compatible
    graph = np.tile(graph.reshape((num_graders, num_assignments, 1)), (1, 1, num_components))
    true_grades = true_grades.reshape((1, num_assignments, num_components))
    reliabilities = reliabilities.reshape((num_graders, 1, 1))
    effort_draws = effort_draws.reshape((num_graders, num_assignments, 1))
    biases = biases.reshape((num_graders, 1, 1))

    # Subset to existing grades
    grader_idx, assignment_idx, component_idx = np.where(graph == 1)
    reported_grades_flat = reported_grades[grader_idx, assignment_idx, component_idx]
    true_grades_flat = true_grades[0, assignment_idx, component_idx]
    reliabilities_flat = reliabilities[grader_idx, 0, 0]
    effort_draws_flat = effort_draws[grader_idx, 0, 0]
    biases_flat = biases[grader_idx, 0, 0]

    # Compute likelihoods
    lower_edges, upper_edges = convert_to_bins(reported_grades_flat, '5')
    high_effort_dist = scipy.stats.norm(loc=true_grades_flat + biases_flat, scale=1/np.sqrt(reliabilities_flat))
    low_effort_dist = scipy.stats.norm(loc=mu_s, scale=1/np.sqrt(tau_l))

    eps = 1e-12
    high_effort_likelihoods = uniform_prob_high_effort / num_bins + (1 - uniform_prob_high_effort) * (high_effort_dist.cdf(upper_edges) - high_effort_dist.cdf(lower_edges))
    low_effort_likelihoods  = uniform_prob_low_effort / num_bins + (1 - uniform_prob_low_effort) * (low_effort_dist.cdf(upper_edges) - low_effort_dist.cdf(lower_edges))
    log_likelihoods_flat = np.log(effort_draws_flat * high_effort_likelihoods + (1 - effort_draws_flat) * low_effort_likelihoods + eps)

    if reduce_per_assignment:
        log_likelihoods_flat = log_likelihoods_flat.reshape(-1, num_components).sum(axis=1)

    return log_likelihoods_flat

    # alternative: 
    # log_likelihoods = np.zeros_like(reported_grades)
    # log_likelihoods[grader_idx, assignment_idx, component_idx] = log_likelihoods_flat
    # return log_likelihoods
    
def lpd(sample_lls):
    """
    Compute the log predictive density of each datapoint, averaged over all posterior samples:
       lpd_i = \log ( \frac{1}{S} \sum_{s=1}^S p(y_i | \theta^s) )
    For details:
    - http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf
    - http://www.stat.columbia.edu/~gelman/research/unpublished/loo_stan.pdf

    Inputs:
    - sample_lls: matrix of (num_samples, num_observations) likelihoods
    
    Returns: array of (num_observations) likelihoods
    """
    # estimate $lpd = \sum_{i=1}^n \log( \frac{1}{S} \sum_{s=1}^S p(y_i | \theta^s) )$ 
    num_samples = len(sample_lls)
    lpd_array = scipy.special.logsumexp(sample_lls, axis=0) - np.log(num_samples)
    return lpd_array

def waic(sample_lls):
    """
    Compute WAIC.
    For details:
    - http://www.stat.columbia.edu/~gelman/research/published/waic_understand3.pdf
    - http://www.stat.columbia.edu/~gelman/research/unpublished/loo_stan.pdf
     
    Inputs:
    - sample_lls: matrix of (num_samples, num_observations) likelihoods
    
    Returns: tuple of
    - elpd: total elpd
    - p_waic: total p_waic estimate
    - elpd_obs: elpd broken down by observation
    - p_waic_obs: p_waic estimate broken down by observation
    """
    sample_lls = sample_lls.astype(float)
    
    # estimate lpd
    lpd_obs = lpd(sample_lls)
    
    # estimate $p_{waic_i} = V_{s=1}^S ( \log p(y_i | \theta^s)
    p_waic_obs = np.var(sample_lls, axis=0, ddof=1)

    # elpd and sums
    elpd_obs = lpd_obs - p_waic_obs
    p_waic = p_waic_obs.sum()
    elpd = elpd_obs.sum()
    
    return (elpd, p_waic, elpd_obs, p_waic_obs)

