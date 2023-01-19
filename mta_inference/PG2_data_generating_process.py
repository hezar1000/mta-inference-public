import numpy as np
import scipy
from scipy import stats
from dataclasses import dataclass

from generator_utils import generate_graph

@dataclass
class PG2Parameters:
    """
    Hyperparameters specific to PG2
    """
    mu_s:         float = None # true grade mean
    sigma_s:      float = None # true grade standard deviation
    mu_b:         float = None # student bias mean (for week 1)
    sigma_b:      float = None # student bias standard deviation
    mu_b_ta:      float = None # TA bias mean (for week 1)
    sigma_b_ta:   float = None # TA bias standard deviation
    alpha_tau:    float = None # alpha param for student reliability gamma distribution
    beta_tau:     float = None # beta param for student reliability gamma distribution
    alpha_tau_ta: float = None # alpha param for TA reliability gamma distribution
    beta_tau_ta:  float = None # beta param for TA reliability gamma distribution

def generate_data(graph_params, PG2_params, seed=None):
    """
    Generate data from PG2 generating process
    """

    if seed is not None:
        np.random.seed(seed)

    # Get graph with no grades filled in yet    
    dataset = generate_graph(graph_params)
    num_graders = dataset.num_graders
    num_submissions = dataset.num_submissions
    grader_roles = dataset.grader_roles
    num_weeks = graph_params.num_weeks

    # Generate true grades
    mu_s = PG2_params.mu_s
    sigma_s = PG2_params.sigma_s
    true_grade_dist = scipy.stats.norm(mu_s, sigma_s)
    true_grades = true_grade_dist.rvs(size=(num_submissions, dataset.num_components))
    dataset.true_grades = true_grades

    # Generate reliabilities (using different distributions for students and TAs)
    alpha_tau = PG2_params.alpha_tau
    alpha_tau_ta = PG2_params.alpha_tau_ta
    beta_tau = PG2_params.beta_tau
    beta_tau_ta = PG2_params.beta_tau_ta
    reliability_alpha = np.array([alpha_tau_ta if role == 'ta' else alpha_tau for role in grader_roles]).reshape(-1, 1)
    reliability_beta = np.array([beta_tau_ta if role == 'ta' else beta_tau for role in grader_roles]).reshape(-1, 1)
    reliabilities = scipy.stats.gamma.rvs(reliability_alpha, scale=1/reliability_beta, size=(num_graders, num_weeks))
    dataset.reliabilities = reliabilities

    # Generate biases (using different distributions for students and TAs)
    mu_b_s = PG2_params.mu_b
    mu_b_ta = PG2_params.mu_b_ta
    sigma_b = PG2_params.sigma_b
    sigma_b_ta = PG2_params.sigma_b_ta

    biases = np.zeros((num_graders, num_weeks))
    bias_sigma_b = np.array([sigma_b_ta if role == 'ta' else sigma_b for role in grader_roles])
    for week_num in range(num_weeks):
        if week_num == 0:
            bias_mu = np.array([mu_b_ta if role == 'ta' else mu_b_s for role in grader_roles])
        else:
            bias_mu = biases[:, week_num - 1]
        biases[:, week_num] = scipy.stats.norm.rvs(bias_mu, bias_sigma_b)
    dataset.biases = biases

    # Generate grades
    observed_grades = np.zeros((num_graders, num_submissions, dataset.num_components))

    for grader_idx in range(num_graders):
        for submission_idx in range(num_submissions):
            week_num = dataset.week_nums[submission_idx]
            if dataset.graph[grader_idx, submission_idx]:
                observed_dist = scipy.stats.norm(true_grades[submission_idx] + biases[grader_idx, week_num],
                                                 1 / np.sqrt(reliabilities[grader_idx, week_num]))
                observed_grade = observed_dist.rvs(size=dataset.num_components)
                observed_grades[grader_idx, submission_idx, :] = observed_grade

    dataset.observed_grades = observed_grades
    dataset.reported_grades = np.around(observed_grades)

    return dataset
