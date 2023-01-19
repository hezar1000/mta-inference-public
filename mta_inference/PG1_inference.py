"""
Binary model, but splitting grades into four components.
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
        Valid keys are 'true_grades', 'reliabilities', 'responsibilities'.
        If key is missing, assume nothing clamped for this variable.
    - num_graders: used to make effort/reliability/responsibility lists, if missing
    - num_assignments: used to make true grade/responsibility lists, if missing
    - num_components: used to make true grades, if missing

    Outputs:
    - tuple of np.ma.arrays for (true grades, efforts, reliabilities, responsibilities)
    """

    # Initialize to default masks of correct shapes
    mask_shapes = {
        'true_grades': (1, num_assignments, num_components),
        'reliabilities': (num_graders, 1, 1),
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
        (mu_s, sigma_s, mu_b, sigma_b, alpha_rel, beta_rel, p_uniform) = hyperparams
        print('No initial point provided; using prior means')
        # True grades are row vector
        true_grades = np.full((1, num_assignments, num_components), mu_s)
        # Reliabilities and efforts are column vectors
        reliabilities = np.full((num_graders, 1, 1), alpha_rel/beta_rel)
        biases = np.full((num_graders, 1, 1), mu_b)

    return true_grades, reliabilities, biases

def run_gibbs(
    observed_grades, graph, hyperparams, 
    initial_point=None, clamped_values={}, max_error=None, num_samples=1000, verbose=True,
):
    """
    Inputs:
    - observed_grades: num_graders x num_assignments x num_components matrix of grades
    - graph: num_graders x num_assignments matrix: 1 if grader v graded assignment u; 0 otherwise
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
    (mu_s, sigma_s, mu_b, sigma_b, alpha_rel, beta_rel, p_uniform) = hyperparams
    tau_s = 1 / sigma_s**2
    eta_0 = 1 / sigma_b**2

    # Set up initial point
    (num_graders, num_assignments, num_components) = observed_grades.shape
    true_grades, reliabilities, biases = set_up_initial_point(hyperparams, num_graders, num_assignments, num_components,
                                                              initial_point)
    graph = graph.reshape(num_graders, num_assignments, 1)

    # Get masks for clamped values
    (true_grades_clamped, reliabilities_clamped) = get_clamp_masks(
        clamped_values, num_graders, num_assignments, num_components    
    )

    # Set up return values: index by (sample number, grader/assignment number, component number [for true grades])
    ret_true_grades = np.zeros((num_samples, num_assignments, num_components))
    ret_biases = np.zeros((num_samples, num_graders))
    ret_reliabilities = np.zeros((num_samples, num_graders))

    for t in tqdm(range(num_samples), disable=not verbose):
        # Sample true grades
        # observed_grade = true_grade + bias
        # true_grade = observed_grade - bias
        # bias = observed_grade - true_grade
        true_grades_prec = (graph * reliabilities).sum(axis=0, keepdims=True) + tau_s
        true_grades_num  = (graph * reliabilities * (observed_grades - biases)).sum(axis=0, keepdims=True) + tau_s * mu_s
        true_grades_mean = true_grades_num / true_grades_prec
        true_grades = stats.norm.rvs(true_grades_mean, 1 / np.sqrt(true_grades_prec), size=(1, num_assignments, num_components))
        true_grades[~true_grades_clamped.mask] = true_grades_clamped[~true_grades_clamped.mask]

        # Sample biases
        biases_prec = (graph * reliabilities).sum(axis=1, keepdims=True)*num_components + eta_0
        biases_num = (graph * reliabilities * (observed_grades - true_grades)).sum(axis=(1,2), keepdims=True) + eta_0 * mu_b
        biases_mean = biases_num / biases_prec
        biases = stats.norm.rvs(biases_mean, 1 / np.sqrt(biases_prec),
                                     size=(num_graders, 1, 1))

        # Sample reliabilities
        grade_errors = (observed_grades - (true_grades + biases))**2
        if max_error is not None:
            grade_errors = np.minimum(grade_errors, max_error**2)
        total_grade_errors = grade_errors.sum(axis=2, keepdims=True)
        reliabilities_alpha = graph.sum(axis=1, keepdims=True)*2 + alpha_rel
        reliabilities_beta = (graph * total_grade_errors).sum(axis=1, keepdims=True)/2 + beta_rel
        reliabilities = stats.gamma.rvs(reliabilities_alpha, scale=1./reliabilities_beta, size=(num_graders, 1, 1))
        reliabilities[~reliabilities_clamped.mask] = reliabilities_clamped[~reliabilities_clamped.mask]

        # Save samples
        ret_true_grades[t, :, :] = true_grades.reshape(num_assignments, num_components)
        ret_reliabilities[t, :] = reliabilities.flatten()
        ret_biases[t, :] = biases.flatten()

    return (ret_true_grades, ret_biases, ret_reliabilities)
