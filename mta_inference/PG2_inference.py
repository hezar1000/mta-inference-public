"""
Binary model, but splitting grades into four components.
"""

from scipy import stats
import numpy as np
from tqdm import tqdm

from inference_utils import sample_clipped_beta, sample_clipped_gamma

# Avoid rounding issues with probabilities close to 0
epsilon = 1e-12

def get_clamp_masks(clamped_variables, num_graders, num_assignments, num_components, num_weeks):
    """
    Get masked arrays from dict of clamp values.

    Example: 
        clamped = {
            'true_grades': [[4, 4, 3, 3], [3, 5, 5, 4], np.nan, np.nan], 
        }
        masks = get_clamp_masks(clamped, 4, 4)

    Inputs:
    - clamped_variables: dict, with array of values to clamp for each variable (or np.nan for unclamped). 
        Valid keys are 'true_grades', 'reliabilities', 'responsibilities'.
        If key is missing, assume nothing clamped for this variable.
    - num_graders: used to make effort/reliability lists, if missing
    - num_assignments: used to make true grade lists, if missing
    - num_components: used to make true grades, if missing
    - num_weeks: used to make reliabilities

    Outputs:
    - tuple of np.ma.arrays for (true grades, efforts, reliabilities, responsibilities)
    """

    # Initialize to default masks of correct shapes
    mask_shapes = {
        'true_grades': (1, num_assignments, num_components),
        'reliabilities': (num_graders, num_weeks, 1),
    }
    ret = {key: np.ma.masked_invalid(np.full(mask_shapes[key], np.nan, dtype=np.float)) for key in mask_shapes}

    # Replace with real values, if present
    for key in mask_shapes:
        if key not in clamped_variables or clamped_variables[key] is None:
            continue
        clamped_floats = np.array(clamped_variables[key], dtype=np.float)
        ret[key] = np.ma.masked_invalid(clamped_floats).reshape(mask_shapes[key])

    return (
        ret['true_grades'], ret['reliabilities']
    )

# TODO
def set_up_initial_point(hyperparams, num_graders, num_assignments, num_components, initial_point):
    """
    Utility function: get initial point for EM and Gibbs and reshape to correct shapes
    """
    if initial_point is not None:
        raise ValueError('Not implemented.')
    else:
        (mu_s, sigma_s, mu_b, sigma_b, alpha_rel, beta_rel) = hyperparams
        print('No initial point provided; using prior means')
        # True grades are row vector
        true_grades = np.full((1, num_assignments, num_components), mu_s)
        # Reliabilities and efforts are (grader x week) matrices
        reliabilities = np.full((num_graders, 1, 1), alpha_rel/beta_rel)
        biases = np.full((num_graders, 1, 1), mu_b)

    return true_grades, reliabilities, biases

def run_gibbs(
    observed_grades, graph, week_nums, hyperparams, 
    initial_point=None, clamped_values={}, max_error=None, num_samples=1000, verbose=True,
):
    """
    Inputs:
    - observed_grades: num_graders x num_assignments x num_components matrix of grades
    - graph: num_graders x num_assignments matrix: 1 if grader v graded assignment u; 0 otherwise
    - week_nums: num_assignments array: which week each assignment came from
    - hyperparams: tuple of (mu_s, sigma_s, mu_b, sigma_b, alpha_rel, beta_rel)
    - initial_point: tuple of (true grades array, effort array, reliability array), or None to use prior means
    - clamped_values: dict of values to clamp. See get_clamp_masks() for format
    - max_error: maximum error (in points) used in reliability updates. No limit if None (default)
    - num_samples: number of samples to take
    - verbose: if True, show progress bar
    """

    # Cast inputs to numpy arrays
    observed_grades = np.array(observed_grades, dtype=np.float)

    # Unpack + process hyperparams
    num_weeks = np.max(week_nums) + 1
    (mu_s, sigma_s, mu_b, sigma_b, alpha_rel, beta_rel) = hyperparams
    tau_s = 1 / sigma_s**2
    eta_0 = 1 / sigma_b**2

    # Set up initial point
    (num_graders, num_assignments, num_components) = observed_grades.shape
    graph = graph.reshape(num_graders, num_assignments, 1)

    # Get masks for clamped values
    (true_grades_clamped, reliabilities_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments, num_components, num_weeks    
    )

    # Set up return values: index by (sample number, grader/assignment number, component number [for true grades])
    ret_true_grades = np.zeros((num_samples, num_assignments, num_components))
    ret_biases = np.zeros((num_samples, num_graders, num_weeks))
    ret_reliabilities = np.zeros((num_samples, num_graders, num_weeks))

    for week_num in range(num_weeks):
        week_assignments = (week_nums == week_num)
        week_graph = graph[:, week_assignments, :]
        week_grades = observed_grades[:, week_assignments, :]
        true_grades_clamped_week = true_grades_clamped[:, week_assignments, :]
        reliabilities_clamped_week = reliabilities_clamped[:, week_num, :][:, np.newaxis, :]
        num_assignments_week = week_graph.shape[1]

        # set up initial point for this week
        true_grades, reliabilities, biases = set_up_initial_point(hyperparams, num_graders, num_assignments_week, num_components,
                                                                  initial_point)
        if week_num == 0:
            bias_mean = mu_b
        else:
            bias_mean = ret_biases[:, :, week_num - 1].mean(axis=0).reshape((num_graders, 1, 1))

        for t in tqdm(range(num_samples), disable=not verbose):
            # Sample true grades
            # observed_grade = true_grade + bias
            # true_grade = observed_grade - bias
            # bias = observed_grade - true_grade
            true_grades_prec = (week_graph * reliabilities).sum(axis=0, keepdims=True) + tau_s
            true_grades_num  = (week_graph * reliabilities * (week_grades - biases)).sum(axis=0, keepdims=True) + tau_s * mu_s
            true_grades_mean = true_grades_num / true_grades_prec
            true_grades = stats.norm.rvs(true_grades_mean, 1 / np.sqrt(true_grades_prec), size=(1, num_assignments_week, num_components))
            true_grades[~true_grades_clamped_week.mask] = true_grades_clamped_week[~true_grades_clamped_week.mask]

            # Sample biases
            biases_prec = (week_graph * reliabilities).sum(axis=1, keepdims=True)*num_components + eta_0
            biases_num = (week_graph * reliabilities * (week_grades - true_grades)).sum(axis=(1,2), keepdims=True) + eta_0 * bias_mean
            biases_mean = biases_num / biases_prec
            biases = stats.norm.rvs(biases_mean, 1 / np.sqrt(biases_prec),
                                    size=(num_graders, 1, 1))

            # Sample reliabilities
            grade_errors = (week_grades - (true_grades + biases))**2
            if max_error is not None:
                grade_errors = np.minimum(grade_errors, max_error**2)
            total_grade_errors = grade_errors.sum(axis=2, keepdims=True)
            reliabilities_alpha = week_graph.sum(axis=1, keepdims=True)*2 + alpha_rel
            reliabilities_beta = (week_graph * total_grade_errors).sum(axis=1, keepdims=True)/2 + beta_rel
            reliabilities = stats.gamma.rvs(reliabilities_alpha, scale=1./reliabilities_beta, size=(num_graders, 1, 1))
            reliabilities[~reliabilities_clamped_week.mask] = reliabilities_clamped_week[~reliabilities_clamped_week.mask]

            # Save samples
            ret_true_grades[t, week_assignments, :] = true_grades.reshape(num_assignments_week, num_components)
            ret_reliabilities[t, :, week_num] = reliabilities.flatten()
            ret_biases[t, :, week_num] = biases.flatten()

    return (ret_true_grades, ret_biases, ret_reliabilities)