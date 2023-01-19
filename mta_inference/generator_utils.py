"""
Helper functions:
- parameters for multiple data generating processes
- containers for generated data
"""

import numpy as np
from dataclasses import dataclass
import bz2
import pickle
import _pickle as cPickle

from mta_inference.matchings import get_random_matching

@dataclass
class GraphParams:
    """
    Inputs shared by all data generating processes.

    TODO: other useful arguments?
    - matching method?
    - non-random TA spotchecks?
    - list of number of calibrations each week? (ex: calibrations_per_week = [18, 10, 8, 5, 5, ...])
    - separate number of calibrations per week for students and TAs?
    """
    # regular grades
    num_components: int = 4     # number of components per grade
    num_students: int = None    # number of student graders (and assignments to simulate)
    num_weeks: int = None       # number of weeks to simulate
    grades_per_week: int = None # number of assignments each student grades per week

    # TAs
    num_tas: int = 0            # number of TA graders
    ta_grades_per_week: int = 0 # number of assignments collectively graded by TA team each week

    # calibrations
    calibrations_per_week: int = 0 # number of calibrations graded by all students/TAs each week

def generate_graph(graph_params):
    """
    Generate the graph for a dataset
    """
    num_students = graph_params.num_students
    num_tas = graph_params.num_tas
    grader_roles = ['student'] * num_students + ['ta'] * num_tas
    num_graders = num_students + num_tas
    num_weeks = graph_params.num_weeks
    week_nums = []
    week_graphs = []
    calibrations = []

    for week_num in range(num_weeks):
        # Calibrations: every grader grades every calibration
        graph_calibrations = np.ones((num_graders, graph_params.calibrations_per_week))
        
        # Student grades: find random matching
        graph_students = np.zeros((num_students, num_students))
        (matching, attempts) = get_random_matching(num_students, graph_params.grades_per_week)
        for (grader, student) in matching:
            graph_students[grader, student] = 1

        # TA spotchecks: shuffle all submissions, then grade first k
        # Assign to TAs by cycling through list of TAs
        graph_spotchecks = np.zeros((num_tas, num_students))
        priorities = np.random.permutation(num_students)[:graph_params.ta_grades_per_week]
        for i, submission_idx in enumerate(priorities):
            graph_spotchecks[i % num_tas, submission_idx] = 1  

        # Build graph for this week: stack student grades on top of TA spotchecks, then calibrations to the left
        graph_submissions = np.vstack([graph_students, graph_spotchecks])
        graph_week = np.hstack([graph_calibrations, graph_submissions])

        week_graphs.append(graph_week)
        week_nums += [week_num] * graph_week.shape[1]
        calibrations += [1] * graph_calibrations.shape[1] + [0] * graph_submissions.shape[1]

    graph = np.hstack(week_graphs)
    week_nums = np.array(week_nums)
    calibrations = np.array(calibrations)

    num_graders, num_submissions = graph.shape

    # Build ordered graph: assume submissions graded in order from left to right
    ordered_graph = np.full_like(graph, -1, dtype=int)
    for grader_idx in range(num_graders):
        grader_count = 0
        for submission_idx in range(num_submissions):
            if graph[grader_idx, submission_idx] == 1:
                ordered_graph[grader_idx, submission_idx] = grader_count
                grader_count += 1

    return GeneratedDataset(
        num_graders = num_graders,
        num_submissions = num_submissions,
        num_components = graph_params.num_components,
        week_nums = week_nums,
        calibrations = calibrations,
        grader_roles = grader_roles,
        graph = graph,
        ordered_graph = ordered_graph
    ) 

def convert_graph_weekly(dataset):
    graph = dataset.graph
    week_nums = dataset.week_nums

    for grader_idx in range(dataset.num_graders):
        for submission_idx in range(dataset.num_submissions):
            if graph[grader_idx, submission_idx]:
                dataset.ordered_graph[grader_idx, submission_idx] = week_nums[submission_idx]
    return dataset

@dataclass
class GeneratedDataset:
    """
    Class for storing outputs of data generating process
    """
    model: str = '' # name of the generating model (e.g. "PG1")

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
    biases: np.ndarray = None          # (num_graders) array of biases for PG1; (num_graders x num_weeks) for PG2
    efforts: np.ndarray = None         # (num_graders) array of effort probabilities
    effort_draws: np.ndarray = None    # (num_graders x num_submissions) matrix of effort draws
    observed_grades: np.ndarray = None # (num_graders x num_submissions x num_components) array of observed grades
    reported_grades: np.ndarray = None # (num_graders x num_submissions x num_components) array of reported grades (for censored models)

def save_dataset(dataset, fname):
	with open(fname, 'wb') as f:
		pickle.dump(dataset, f)

def load_dataset(fname):
	with open(fname, 'rb') as f:
		return pickle.load(f)

def save_compressed_dataset(dataset, fname):
    with bz2.BZ2File(fname + '.pbz2', 'w') as f: 
         cPickle.dump(dataset, f)

def load_compressed_dataset(fname):
    data = bz2.BZ2File(fname, 'rb')
    data = cPickle.load(data)
    return data
