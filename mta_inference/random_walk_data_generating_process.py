import numpy as np
import scipy
from scipy import stats
from dataclasses import dataclass

from generator_utils_modified import generate_graph

@dataclass
class RandomWalkModelParams:
    """
    Hyperparameters specific to the component model
    """
    mu_s:         float = None # true grade mean
    sigma_s:      float = None # true grade standard deviation
    
    alpha_e:      float = None # alpha param for effort beta distribution
    beta_e:       float = None # beta param for effort beta distribution

    alpha_tau:    float = None # alpha param for student reliability gamma distribution
    beta_tau:     float = None # beta param for student reliability gamma distribution
    alpha_aux:    float = None # alpha param for student auxiliary gamma distribution
    beta_aux:     float = None # beta param for student auxiliary gamma distribution

    alpha_tau_ta: float = None # alpha param for TA reliability gamma distribution
    beta_tau_ta:  float = None # beta param for TA reliability gamma distribution
    alpha_aux_ta: float = None # alpha param for TA auxiliary gamma distribution
    beta_aux_ta:  float = None # beta param for TA auxiliary gamma distribution
    
    tau_l:        float = None # low effort reliability

def generate_data(graph_params, component_model_params, seed=None):
    """
    Generate data from component model generating process
    """

    if seed is not None:
        np.random.seed(seed)

    # Get graph with no grades filled in yet    
    dataset = generate_graph(graph_params)
    num_graders = dataset.num_graders
    num_submissions = dataset.num_submissions
    grader_roles = dataset.grader_roles
    grades_per_grader = dataset.ordered_graph.max(axis=1)

    # Generate true grades
    mu_s = component_model_params.mu_s
    sigma_s = component_model_params.sigma_s
    true_grade_dist = scipy.stats.norm(mu_s, sigma_s)
    true_grades = true_grade_dist.rvs(size=(num_submissions, dataset.num_components))
    dataset.true_grades = true_grades

    # Generate efforts (but modify TA efforts to 1.0)
    alpha_e = component_model_params.alpha_e
    beta_e = component_model_params.beta_e
    efforts = scipy.stats.beta.rvs(alpha_e, beta_e, size=num_graders)
    efforts[np.array(grader_roles) == 'ta'] = 1.0
    dataset.efforts = efforts

    # Generate reliability time series (using different distributions for students and TAs)
    reliabilities = []
    for grader_idx in range(num_graders):
        alpha_tau = component_model_params.alpha_tau_ta if grader_roles[grader_idx] == 'ta' else component_model_params.alpha_tau
        beta_tau  = component_model_params.beta_tau_ta  if grader_roles[grader_idx] == 'ta' else component_model_params.beta_tau
        alpha_aux = component_model_params.alpha_aux_ta if grader_roles[grader_idx] == 'ta' else component_model_params.alpha_aux
        beta_aux  = component_model_params.beta_aux_ta  if grader_roles[grader_idx] == 'ta' else component_model_params.beta_aux

        auxiliary_grader = [0] * (grades_per_grader[grader_idx] + 1)
        reliabilities_grader = [0] * (grades_per_grader[grader_idx] + 1)

        for t in range(grades_per_grader[grader_idx] + 1):
            if t == 0:
                auxiliary_grader[t] = 1.0
            else:
                auxiliary_grader[t] = scipy.stats.gamma.rvs(alpha_aux, scale=1/(beta_aux * reliabilities_grader[t-1]))

            reliabilities_grader[t] = scipy.stats.gamma.rvs(alpha_tau, scale=1/(beta_tau * auxiliary_grader[t]))
        reliabilities.append(reliabilities_grader)
    dataset.reliabilities = reliabilities

    # Generate grades
    effort_draws = np.zeros((num_graders, num_submissions))
    observed_grades = np.zeros((num_graders, num_submissions, dataset.num_components))

    tau_l = component_model_params.tau_l
    low_effort_dist = scipy.stats.norm(mu_s, 1 / np.sqrt(tau_l))

    for grader_idx in range(num_graders):
        for submission_idx in range(num_submissions):
            if dataset.graph[grader_idx, submission_idx]:
                effort_draw = scipy.stats.bernoulli.rvs(efforts[grader_idx])
                if effort_draw:
                    grader_time_step = dataset.ordered_graph[grader_idx, submission_idx]
                    observed_dist = scipy.stats.norm(true_grades[submission_idx], 1 / np.sqrt(reliabilities[grader_idx][grader_time_step]))
                else:
                    observed_dist = low_effort_dist
                observed_grade = observed_dist.rvs(size=dataset.num_components)

                effort_draws[grader_idx, submission_idx] = effort_draw
                observed_grades[grader_idx, submission_idx, :] = observed_grade

    dataset.effort_draws = effort_draws
    dataset.observed_grades = observed_grades
    dataset.reported_grades = np.around(observed_grades)

    return dataset

def generate_data_weekly(graph_params, component_model_params, seed=None):
    """
    Generate data from random walk generating process.
    Assumes only one random walk update for each week.
    """

    if seed is not None:
        np.random.seed(seed)

    # Get graph with no grades filled in yet    
    dataset = generate_graph(graph_params)
    dataset = convert_graph_weekly(dataset)

    num_graders = dataset.num_graders
    num_submissions = dataset.num_submissions
    grader_roles = dataset.grader_roles
    num_weeks = graph_params.num_weeks

    # Generate true grades
    mu_s = component_model_params.mu_s
    sigma_s = component_model_params.sigma_s
    true_grade_dist = scipy.stats.norm(mu_s, sigma_s)
    true_grades = true_grade_dist.rvs(size=(num_submissions, dataset.num_components))
    dataset.true_grades = true_grades

    # Generate efforts (but modify TA efforts to 1.0)
    alpha_e = component_model_params.alpha_e
    beta_e = component_model_params.beta_e
    efforts = scipy.stats.beta.rvs(alpha_e, beta_e, size=num_graders)
    efforts[np.array(grader_roles) == 'ta'] = 1.0
    dataset.efforts = efforts

    # Generate reliability time series (using different distributions for students and TAs)
    reliabilities = np.zeros((num_graders, num_weeks))
    for grader_idx in range(num_graders):
        alpha_tau = component_model_params.alpha_tau_ta if grader_roles[grader_idx] == 'ta' else component_model_params.alpha_tau
        beta_tau  = component_model_params.beta_tau_ta  if grader_roles[grader_idx] == 'ta' else component_model_params.beta_tau
        alpha_aux = component_model_params.alpha_aux_ta if grader_roles[grader_idx] == 'ta' else component_model_params.alpha_aux
        beta_aux  = component_model_params.beta_aux_ta  if grader_roles[grader_idx] == 'ta' else component_model_params.beta_aux

        for t in range(num_weeks):
            if t == 0:
                aux = 1.0
            else:
                aux = scipy.stats.gamma.rvs(alpha_aux, scale=1/(beta_aux * reliabilities[grader_idx, t-1]))

            reliabilities[grader_idx, t] = scipy.stats.gamma.rvs(alpha_tau, scale=1/(beta_tau * aux))
    dataset.reliabilities = reliabilities

    # Generate grades
    effort_draws = np.zeros((num_graders, num_submissions))
    observed_grades = np.zeros((num_graders, num_submissions, dataset.num_components))

    tau_l = component_model_params.tau_l
    low_effort_dist = scipy.stats.norm(mu_s, 1 / np.sqrt(tau_l))

    for grader_idx in range(num_graders):
        for submission_idx in range(num_submissions):
            week_num = dataset.week_nums[submission_idx]
            if dataset.graph[grader_idx, submission_idx]:
                effort_draw = scipy.stats.bernoulli.rvs(efforts[grader_idx])
                if effort_draw:
                    observed_dist = scipy.stats.norm(true_grades[submission_idx], 1 / np.sqrt(reliabilities[grader_idx, week_num]))
                else:
                    observed_dist = low_effort_dist
                observed_grade = observed_dist.rvs(size=dataset.num_components)

                effort_draws[grader_idx, submission_idx] = effort_draw
                observed_grades[grader_idx, submission_idx, :] = observed_grade

    dataset.effort_draws = effort_draws
    dataset.observed_grades = observed_grades
    dataset.reported_grades = np.around(observed_grades)

    return dataset
