# Helper functions:
# - containers for generated data
# - functions that are useful for Gibbs/EM in multiple models
# - functions for post-processing inference results
import numpy as np
import scipy.stats 
from dataclasses import dataclass
from bisect import bisect_left


@dataclass
class GeneratedDataset:
    """
    Class for storing outputs of data generating process
    """
    num_graders: int = None         # number of students + TAs
    num_submissions: int = None     # total number of assignments 
    num_components: int = None      # number of components in each grade
    week_nums: np.ndarray = None    # (num_submissions) array of week number for each submission
    calibrations: np.ndarray = None # (num_submissions) array: 1 if submission is calibration
    grader_roles: list = None       # (num_graders) list of 'student' or 'ta'

    graph: np.ndarray = None         # (num_graders x num_submissions) matrix: 1 if grader graded submission
    ordered_graph: np.ndarray = None # (num_graders x num_submissions) matrix: order of grades for each grader

    true_grades: np.ndarray = None     # (num_submissions x num_components) array of grades
    reliabilities: list = None         # (num_graders) list of dependabilities; list of lists for random walk models
    efforts: np.ndarray = None         # (num_graders) array of effort probabilities
    effort_draws: np.ndarray = None    # (num_graders x num_submissions) matrix of effort draws
    observed_grades: np.ndarray = None # (num_graders x num_submissions x num_components) array of observed grades
    reported_grades: np.ndarray = None # (num_graders x num_submissions x num_components) array of reported grades (for censored models)

def convertGradeScale25To5(grades):
    """
    Convert 25-point grades to 5-point grades
    """
    for (old_grade, new_grade) in [(25, 5), (20, 4), (16.25, 3), (12.5, 2), (6.25, 1)]:
        grades[np.where(grades == old_grade)] = new_grade

    return grades

# mapping from reported grades to (left edge, right edge)
def generate_bins(bin_centers, left_edge, right_edge):
    bins = {}
    for i in range(len(bin_centers)):
        if i == 0:
            bin_left_edge = left_edge
        else:
            bin_left_edge = (bin_centers[i-1] + bin_centers[i]) / 2

        if i+1 == len(bin_centers):
            bin_right_edge = right_edge
        else:
            bin_right_edge = (bin_centers[i] + bin_centers[i+1]) / 2 
        bins[bin_centers[i]] = (bin_left_edge, bin_right_edge)
    return bins

def convert_to_bins(reported_grades, grade_bins):
    def get_lower_edge(reported_grade):
        return grade_bins[reported_grade][0]

    def get_upper_edge(reported_grade):
        return grade_bins[reported_grade][1]

    left_edges = np.vectorize(get_lower_edge)(reported_grades)
    right_edges = np.vectorize(get_upper_edge)(reported_grades)
    return left_edges, right_edges

def set_up_initial_point_PG5(hyperparams, num_graders, num_assignments, num_components, initial_point=None, verbose=False):
    """
    Utility function: get initial point for EM and Gibbs and reshape to correct shapes
    """

    if initial_point is None:
        if verbose:
            print('No initial point provided; using prior means')
        true_grades = np.full((1, num_assignments, num_components), hyperparams['mu_s'])
        reliabilities = np.full((num_graders, 1, 1), hyperparams['mu_s'])
        efforts = np.full((num_graders, 1, 1), hyperparams['alpha_e'] / (hyperparams['alpha_e'] + hyperparams['beta_e']))
        effort_draws = np.full((num_graders, num_assignments, 1), 1)
        biases = np.full((num_graders, 1, 1), 0)

    else:
        raise NotImplementedError("PG5 does not support initial points other than the prior means")

    return {
        'true_grades': true_grades, 
        'reliabilities': reliabilities, 
        'biases': biases,
        'efforts': efforts,
        'effort_draws': effort_draws,
    }


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
        Valid keys are 'true_grades', 'biases', 'efforts', 'reliabilities', 'effort_draws'.
        If key is missing, assume nothing clamped for this variable.
    - num_graders: used to make effort/reliability/responsibility lists, if missing
    - num_assignments: used to make true grade/responsibility lists, if missing
    - num_components: used to make true grades, if missing

    Outputs:
    - dictionary of np.ma.arrays
    """

    # Initialize to default masks of correct shapes
    mask_shapes = {
        'true_grades': (1, num_assignments, num_components),
        'reliabilities': (num_graders, 1, 1),
        'biases': (num_graders, 1, 1),
        'efforts': (num_graders, 1, 1),
        'effort_draws': (num_graders, num_assignments, 1)
    }
    ret = {key: np.ma.masked_invalid(np.full(mask_shapes[key], np.nan, dtype=np.float)) for key in mask_shapes}

    # Replace with real values, if present
    for key in mask_shapes:
        if key not in clamped_variables or clamped_variables[key] is None:
            continue
        clamped_floats = np.array(clamped_variables[key], dtype=np.float)
        ret[key] = np.ma.masked_invalid(clamped_floats).reshape(mask_shapes[key])

    return ret

def clamp_parameters(current_sample, clamped_values, variable=None):
    """
    Modify the current parameters stored in current_sample.

    Inputs:
    - current_sample: dictionary of np arrays with parameter values
    - clamped_values: dictionary of np masked arrays with clamped values
    - variable: the name of a variable to clamp. if none, clamp all variables in clamped_values.

    Returns: updated copy of current_sample
    """

    # Clamp all variables
    if variable is None:
        for var_name in clamped_values:
            current_sample = clamp_parameters(current_sample, clamped_values, variable=var_name)

    # Clamp single variable
    else:
        current_sample[variable][~clamped_values[variable].mask] = clamped_values[variable][~clamped_values[variable].mask]

    return current_sample

def norm_cdf(x, mean, precision):
    # equivalent to scipy.stats.norm.cdf(x, mean, 1/np.sqrt(precision))
    x_transformed = np.sqrt(precision) * (x - mean)
    return scipy.special.ndtr(x_transformed)  

def sample_categorical(x, p):
    """
    Sample from each row of x according to p
    
    Inputs: 
    - x: n-dimensional matrix of possible values for x
    - p: n-dimensional matrix of probabilities for each sample; p.sum(axis=-1) must be all-1s
    
    Outputs:
    - (n-1)-dimensional array of samples
    """

    # Convert x and p to 2D
    shape_2D = (np.prod(x.shape[:-1]), x.shape[-1])
    x_2D = x.reshape(shape_2D)
    p_2D = p.reshape(shape_2D)

    # Sample from each row
    p_cumulative = np.cumsum(p_2D, axis=1)
    quantiles = np.random.uniform(size=len(p_2D))
    samples = [x_2D[row, np.searchsorted(p_cumulative[row], quantiles[row])] for row in range(len(p_2D))]
    # samples = [np.random.choice(x[row], p=p[row]) for row in range(len(x))]
    return np.array(samples).reshape(x.shape[:-1])

def convert_ll_to_prob(log_likelihoods):
    """
    Convert unnormalized log-likelihoods to probabilities

    Inputs:
    - log_likelihoods: n-dimensional matrix of log-likelihoods

    Outputs:
    - n-dimensional matrix of probabilities; sum across final dimension is always equal to 1
    """
    scaled_likelihoods = log_likelihoods - np.max(log_likelihoods, axis=-1, keepdims=True)
    p_unnorm = np.exp(scaled_likelihoods)
    return p_unnorm / p_unnorm.sum(axis=-1, keepdims=True)

def get_ci(samples, ci):
    """
    Get a confidence interval from samples.
    Example for 95% confidence interval:
        samples = np.random.random(100)
        ci_min, ci_max = get_ci(samples, 0.95)

    Inputs:
    - samples: list or numpy array of data
    - ci: fraction of mass to include in interval. 
          Resulting interval ranges from the (1-ci)/2th to (1+ci)/2th percentile

    Outputs:
    - Tuple of (ci_min, ci_max)
    """
    
    # Note: percentiles are numbers in [0, 100]
    lower_percentile = 100 * (1 - ci) / 2
    upper_percentile = 100 * (1 + ci) / 2
    [ci_min, ci_max] = np.percentile(samples, [lower_percentile, upper_percentile])
    return (ci_min, ci_max)

def convert_dependability_to_grade(dependability, keypoints):
    """
    Translate dependabilities into a grade by linearly interpolating between some keypoints.
    
    Inputs:
    - dependability: a single student's dependability (float) or a list of dependabilities
    - keypoints: a list of (dependability, grade) pairs to define grading curve
    
    Output:
    - a single grade or list of grades, depending on input format
    """
    
    # Split keypoints into x and y coords
    xp = [kp[0] for kp in keypoints]
    yp = [kp[1] for kp in keypoints]
    
    # Interpolate
    grade = np.interp(dependability, xp, yp)
    
    return grade

# TODO: (re)move some of these internal helper functions
def sample_clipped_distribution(dist, size=None, min_val=None, max_val=None):
    """
    Helper function: draw samples x ~ dist, subject to min_val <= x <= max_val.
    Uses inverse CDF sampling.
    """

    # Uniformly sample quantiles
    epsilon = 1e-10

    min_cdf = 0 if min_val is None else dist.cdf(min_val)
    max_cdf = 1 if max_val is None else dist.cdf(max_val)
    # min_cdf = np.clip(0 if min_val is None else dist.cdf(min_val), epsilon, 1-epsilon)
    # max_cdf = np.clip(1 if max_val is None else dist.cdf(max_val), epsilon, 1-epsilon)
    quantiles = scipy.stats.uniform.rvs(min_cdf, max_cdf-min_cdf, size=size)

    # Transform into samples
    samples = dist.ppf(quantiles)
    if max_val is not None:
        samples[samples > max_val] = max_val[samples > max_val]
    if min_val is not None:
        samples[samples < min_val] = min_val[samples < min_val]

    return samples

def sample_clipped_beta(alpha, beta, size=None, min_val=None, max_val=None):
    """
    Draw samples x ~ Beta(alpha, beta), subject to min_val <= x <= max_val.
    
    All inputs can either be scalar value (float) or numpy array.

    Inputs:
    - alpha: alpha parameter for beta distribution
    - beta: beta parameter for beta distribution
    - size: shape of sample array
    - min_val: lower bound on samples; no lower bound if None
    - max_val: upper bound on samples; no upper bound if None
    """

    dist = scipy.stats.beta(alpha, beta)
    return sample_clipped_distribution(dist, size, min_val, max_val)

def sample_clipped_gamma(alpha, beta, size=None, min_val=None, max_val=None):
    """
    Draw samples x ~ Gamma(alpha, beta), subject to min_val <= x <= max_val.
    
    All inputs can either be scalar value (float) or numpy array.

    Inputs:
    - alpha: alpha parameter for beta distribution
    - beta: beta parameter for beta distribution
    - size: shape of sample array
    - min_val: lower bound on samples; no lower bound if None
    - max_val: upper bound on samples; no upper bound if None
    """

    dist = scipy.stats.gamma(alpha, scale=1/beta)
    return sample_clipped_distribution(dist, size, min_val, max_val)

def sample_clipped_normal(mean, precision, size=None, min_val=None, max_val=None):
    """
    Draw samples x ~ Gamma(alpha, beta), subject to min_val <= x <= max_val.
    
    All inputs can either be scalar value (float) or numpy array.

    Inputs:
    - alpha: alpha parameter for beta distribution
    - beta: beta parameter for beta distribution
    - size: shape of sample array
    - min_val: lower bound on samples; no lower bound if None
    - max_val: upper bound on samples; no upper bound if None
    """

    dist = scipy.stats.norm(mean, 1/np.sqrt(precision))
    return sample_clipped_distribution(dist, size, min_val, max_val)

def truncnorm_stats(mu, std, a, b):
    """
    Find the mean and variance of truncnorm(mu, std, a, b)
    (x ~ N(mean, std), subject to a <= x <= b)
    """
    alpha = (a - mu) / std
    beta = (b - mu) / std
    phi_alpha = scipy.stats.norm.pdf(alpha)
    phi_beta = scipy.stats.norm.pdf(beta)
    Z = scipy.stats.norm.cdf(beta) - scipy.stats.norm.cdf(alpha)

    mean = mu + std * (phi_alpha - phi_beta) / Z
    var = std**2 * (1 + (alpha*phi_alpha - beta*phi_beta) / Z - ((phi_alpha - phi_beta) / Z)**2)

    return mean, var

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def mae(predictions, targets):
    return np.abs(predictions - targets).sum()

# TODO: this appears broken. implement round_nearest.
def convert_observed_grades_to_bins(observed_grades, bin_width, min_value=0.0, max_value=5.0, large_value=20):
    bin_centers = round_nearest(observed_grades, bin_width)
    bin_centers[bin_centers < min_value] = min_value
    bin_centers[bin_centers > max_value] = max_value

    bin_left_edges = bin_centers - bin_width / 2
    bin_left_edges[bin_centers == min_value] = -large_value

    bin_right_edges = bin_centers + bin_width / 2
    bin_right_edges[bin_centers == max_value] = large_value

    return bin_left_edges, bin_right_edges

def make_validation_sets(
    observed_grades, graph, calibration_grades, effort_grades, students, submissions, 
    num_splits=10, seed=12345
):
    """
    Use like:

    for dataset in make_validation_sets(observed_grades, graph, calibration_grades, effort_grades, students, submissions, seed=12345):
        # run inference using dataset['observed_grades'] and dataset['graph']
        # here's a placeholder: 
        inferred_true_grades = observed_grades
        
        # calculate errors
        split_error = rmse(
            inferred_true_grades[dataset['target_graders'], dataset['target_assignments']],
            dataset['target_grades']
        )
        print(split_error)
    """
    np.random.seed(seed)
    
    # steps:
    # - identify all non-calibration TA grades
    # (list of grader, assignment pairs where grader is TA and assignment is not calibration)
    student_list = list(students.items())
    submission_list = list(submissions.items())
    
    ta_graph = graph.copy()
    for (grader_idx, assignment_idx) in np.argwhere(ta_graph):
        (grader_num, grader_type) = student_list[grader_idx]
        (assignment_num, assignment_type) = submission_list[assignment_idx]
        
        if grader_type != 'ta' or assignment_type == 'calibration':
            ta_graph[grader_idx, assignment_idx] = 0
            
    ta_grades = np.argwhere(ta_graph)

    # - split into 10 groups at random (shuffle)
    np.random.shuffle(ta_grades)
    split_points = np.linspace(0, len(ta_grades), num_splits+1).astype(int)
    
    for i in range(num_splits):
        # - remove group of TA grades from graph, observed_grades
        observed_grades_split = observed_grades.copy()
        graph_split = graph.copy() 
        
        validation_idx = ta_grades[split_points[i]:split_points[i+1]]
        val_grader_idx = validation_idx[:, 0]
        val_assignment_idx = validation_idx[:, 1]
        val_grades = observed_grades[val_grader_idx, val_assignment_idx]
        
        observed_grades_split[val_grader_idx, val_assignment_idx] = 0
        graph_split[val_grader_idx, val_assignment_idx] = 0
        
        # - return revised dataset and validation targets        
        yield {
            'observed_grades': observed_grades_split,
            'graph': graph_split,
            'target_graders': val_grader_idx, 
            'target_assignments': val_assignment_idx, 
            'target_grades': val_grades,
        }
        
        
def make_calibration_validation_sets(
    observed_grades, graph, calibration_grades, effort_grades, students, submissions, 
    num_splits=10, targets_per_split=100, graders_per_target=4, valid_submissions=None,
    seed=12345
):
    """
    Make validation sets by sampling grades from calibrations
    """
    
    np.random.seed(seed)
    (num_graders, num_assignments, num_components) = observed_grades.shape
    
    # steps:
    # - identify calibration grades
    submission_list = list(submissions.items())
    calibration_ids = [i for (i, (submission_id, submission_type)) in enumerate(submission_list) if submission_type == 'calibration']
    
    # - filter for valid calibration IDs
    if valid_submissions is not None:
        calibration_ids = [i for i in calibration_ids if i in valid_submissions]
    
    # - for each split:
    for i in range(num_splits):
        observed_grades_target = np.zeros((num_graders, targets_per_split, num_components))
        graph_target = np.zeros((num_graders, targets_per_split))
        grades_target = np.zeros((targets_per_split, num_components))
        
        # - choose calibrations to generate from
        split_calibrations = np.random.choice(calibration_ids, size=targets_per_split, replace=True)
    
        for (i, assignment) in enumerate(split_calibrations):
            # - choose graders for each calibration
            calibration_graders = np.where(graph[:, assignment])[0]
            target_graders = np.random.choice(calibration_graders, size=graders_per_target, replace=False)

            # - add to observed_grades and graph
            for grader in target_graders:
                observed_grades_target[grader, i, :] = observed_grades[grader, assignment, :]
                graph_target[grader, i] = 1
                
            grades_target[i, :] = calibration_grades[assignment, :]
        
        observed_grades_split = np.concatenate([
            observed_grades,
            observed_grades_target,
        ], axis=1)
        graph_split = np.concatenate([
            graph, 
            graph_target,
        ], axis=1)
        target_idx = np.arange(observed_grades.shape[1], observed_grades_split.shape[1])
    
        yield {
            'observed_grades': observed_grades_split,
            'graph': graph_split,
            'target_assignments': target_idx, 
            'target_grades': grades_target,
        }


def take_closest(value, bins):
    """
    Assumes bins is sorted. Returns closest bin to value.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(bins, value)
    if pos == 0:
        return bins[0]
    if pos == len(bins):
        return bins[-1]
    before = bins[pos - 1]
    after = bins[pos]
    if after - value < value - before:
        return after
    else:
        return before

take_closest = np.vectorize(take_closest, otypes=[np.float32], excluded=[1, 'bins'])

def convert_samples_to_bin_histograms(samples, bins):
    binned_samples = take_closest(samples.astype(float), bins)
    bin_probabilities = np.stack([
        np.mean(np.abs(binned_samples - bin_value) < 1e-6, axis=0)
        for bin_value in bins
    ])
    return bin_probabilities

def buildAuthorGraph(graph, submissions):
    author_graph = np.zeros_like(graph)
    try:
        for sub_id, (_, (sub_type, author_id)) in enumerate(submissions.items()):
            if sub_type == 'submission':
                author_graph[int(author_id), sub_id] = 1
    except:
        author_graph = np.zeros_like(graph)         
    return author_graph



