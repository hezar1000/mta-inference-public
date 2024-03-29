import numpy as np
import scipy
from scipy import stats
from dataclasses import dataclass

#from generator_utils import generate_graph
from mta_inference.generator_utils_modified import generate_graph

@dataclass
class PGnModelParams:
    """
    Hyperparameters specific to the component+bias model
    """
    mu_s:         float = None # true grade mean
    sigma_s:      float = None # true grade standard deviation
    mu_b: float = None  # student bias mean
    sigma_b: float = None  # student bias standard deviation
    mu_b_ta: float = None  # TA bias mean
    sigma_b_ta: float = None  # TA bias standard deviation
    alpha_e:      float = None # alpha param for effort beta distribution
    beta_e:       float = None # beta param for effort beta distribution
    alpha_tau:    float = None # alpha param for student reliability gamma distribution
    beta_tau:     float = None # beta param for student reliability gamma distribution
    alpha_tau_ta: float = None # alpha param for TA reliability gamma distribution
    beta_tau_ta:  float = None # beta param for TA reliability gamma distribution
    tau_l:        float = None # low effort reliability
    p_uniform_low_effort:    float = None # p uniform low effort

def generate_data(graph_params, pgn_model_params, seed=None):
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

    # Generate true grades
    mu_s = pgn_model_params.mu_s
    sigma_s = pgn_model_params.sigma_s
    true_grade_dist = scipy.stats.norm(mu_s, sigma_s)
    true_grades = true_grade_dist.rvs(size=(num_submissions, dataset.num_components))
    dataset.true_grades = true_grades
    p_uniform_low_effort = pgn_model_params.p_uniform_low_effort

    # Generate efforts (but modify TA efforts to 1.0)
    alpha_e = pgn_model_params.alpha_e
    beta_e = pgn_model_params.beta_e
    efforts = scipy.stats.beta.rvs(alpha_e, beta_e, size=num_graders)
    efforts[np.array(grader_roles) == 'ta'] = 1.0
    dataset.efforts = efforts

    # Generate reliabilities (using different distributions for students and TAs)
    alpha_tau = pgn_model_params.alpha_tau
    alpha_tau_ta = pgn_model_params.alpha_tau_ta
    beta_tau = pgn_model_params.beta_tau
    beta_tau_ta = pgn_model_params.beta_tau_ta
    reliability_alpha = np.array([alpha_tau_ta if role == 'ta' else alpha_tau for role in grader_roles])
    reliability_beta = np.array([beta_tau_ta if role == 'ta' else beta_tau for role in grader_roles])
    reliabilities = scipy.stats.gamma.rvs(reliability_alpha, scale=1/reliability_beta)
    dataset.reliabilities = reliabilities

    # Generate biases (using different distributions for students and TAs)
    mu_b_s = pgn_model_params.mu_b
    mu_b_ta = pgn_model_params.mu_b_ta
    sigma_b = pgn_model_params.sigma_b
    sigma_b_ta = pgn_model_params.sigma_b_ta
    bias_mu = np.array([mu_b_ta if role == 'ta' else mu_b_s for role in grader_roles])
    bias_sigma_b = np.array([sigma_b_ta if role == 'ta' else sigma_b for role in grader_roles])
    biases = scipy.stats.norm.rvs(bias_mu, bias_sigma_b)
    dataset.biases = biases

    # Generate grades
    effort_draws = np.zeros((num_graders, num_submissions))
    observed_grades = np.zeros((num_graders, num_submissions, dataset.num_components))

    tau_l = pgn_model_params.tau_l
    
    uniform_low_effort_draw = scipy.stats.bernoulli.rvs(1-p_uniform_low_effort)
    if uniform_low_effort_draw:
        low_effort_dist = scipy.stats.norm(mu_s, 1 / np.sqrt(tau_l))
    else:
        low_effort_dist = scipy.stats.uniform(loc=-0.5, scale = 5.5)

    for grader_idx in range(num_graders):
        for submission_idx in range(num_submissions):
            if dataset.graph[grader_idx, submission_idx]:
                effort_draw = scipy.stats.bernoulli.rvs(efforts[grader_idx])
                if effort_draw:
                    observed_dist = scipy.stats.norm(true_grades[submission_idx]+ biases[grader_idx], 1 / np.sqrt(reliabilities[grader_idx]))
                else:
                    observed_dist = low_effort_dist

                        
                observed_grade = observed_dist.rvs(size=dataset.num_components)
                effort_draws[grader_idx, submission_idx] = effort_draw
                observed_grades[grader_idx, submission_idx, :] = observed_grade

    dataset.effort_draws = effort_draws
    dataset.observed_grades = observed_grades
    dataset.reported_grades = np.around(observed_grades)

    return dataset

"""
TODO: Helpful to have a command line interface, too?
If so, here's a start...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for component model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results to')
    # graph structure arguments
    parser.add_argument('--num_components',        type=int, default=4,     help='Number of components per grade')
    parser.add_argument('--num_weeks',             type=int, required=True, help='Number of weeks to simulate')
    parser.add_argument('--num_students',          type=int, required=True, help='Number of student graders (and assignments to be graded each week)')
    parser.add_argument('--grades_per_week',       type=int, required=True, help='Number of assignments each student grades per week')
    parser.add_argument('--num_tas',               type=int, default=0,     help='Number of TA graders')
    parser.add_argument('--ta_grades_per_week',    type=int, default=0,     help='Number of assignments the TAs collectively grade per week')
    parser.add_argument('--calibrations_per_week', type=int, default=0,     help='Number of calibrations graded each week')
    # distribution hyperparameters
    parser.add_argument('--mu_s',    type=float, required=True, help='True grade mean')
    parser.add_argument('--sigma_s', type=float, required=True, help='True grade standard deviation')
    parser.add_argument('--alpha_e')
    parser.add_argument('--beta_e')
    parser.add_argument('--alpha_tau')
    parser.add_argument('--beta_tau')
    parser.add_argument('--alpha_tau_ta')
    parser.add_argument('--beta_tau_ta')
    parser.add_argument('--tau_l')
"""
