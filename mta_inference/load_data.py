import numpy as np
import pandas as pd
from collections import OrderedDict
# from walrus import *


def load_mta_data(fnames, excluded_submissions=[], verbose=True):
    """
    Load data from MTA

    Input:
    - fnames: list of paths to CSV files
    - excluded_submissions: list of review IDs to exclude from dataset

    Output:
    - observed_grades: 3D matrix of grades
    - graph: 2D binary matrix showing which grades are legitimate
    - calibration_grades: 3D matrix of true grades; equal to np.nan for regular assignments, and true values for calibrations
    - effort_grades: 2D matrix of responsibilities; equal to 0 or 1 when graded by a TA, or np.nan if not
    - grader_list: OrderedDict of {student ID: 'student' or 'ta'}
    - submission_list: list of submission IDs
    - df: raw dataframe from concatenating weekly data
    """

    # Load data
    dfs = []
    for (week_num, fname) in enumerate(fnames):
        df_week = pd.read_csv(fname, dtype={'Student ID': str, 'Reviewer ID': str, 'Submission Date': str, 'Deadline': str})
        df_week['Week'] = week_num+1
        dfs.append(df_week)
    df = pd.concat(dfs)

    # Apply other processing
    df['Reviewer Role'] = df['Reviewer Role'].apply(lambda s: s.strip() if type(s) == str else s)
    df['Student ID'] = df['Student ID'].apply(lambda s: s.strip() if type(s) == str else s)
    df['Reviewer ID'] = df['Reviewer ID'].apply(lambda s: s.strip() if type(s) == str else s)

    # Get student list and assignment list
    # Note: misses any students who graded 0 assignments
    grader_lookup = OrderedDict()
    for role in ['student', 'ta', 'instructor']:
        for student in np.unique(df[df['Reviewer Role'] == role]['Reviewer ID']):
            grader_lookup[student] = role
    
    assignment_lookup = OrderedDict()
    for assignment in np.unique(df['Submission ID']):
        assignment_lookup[assignment] = 'submission' # default value

    # Build reverse lookups for speed
    grader_idx_lookup = {student: idx for (idx, student) in enumerate(grader_lookup.keys())}
    assignment_idx_lookup = {assignment: idx for (idx, assignment) in enumerate(assignment_lookup.keys())}

    # Build matrices
    # component_names = ['1. Argument structure', '4. English', '2. Evidence', '3. Subject matter']
    component_names = ['Argument structure', 'English', 'Evidence', 'Subject matter']
    num_graders = len(grader_lookup)
    num_assignments = len(assignment_lookup)
    num_components = len(component_names)

    observed_grades = np.zeros((num_graders, num_assignments, num_components))
    calibration_grades = np.full((num_assignments, num_components), np.nan)
    effort_grades = np.full((num_graders, num_assignments), np.nan)
    graph = np.zeros((num_graders, num_assignments))
    review_dates = np.empty((num_graders, num_assignments), dtype=object)
    review_ids = np.zeros((num_graders, num_assignments))
    week_nums = np.zeros(num_assignments)

    for index, row in df.iterrows():
        grader = row['Reviewer ID']
        assignment = row['Submission ID']
        week = row['Week']
        grades = [row[component] for component in component_names]
        review_date = row['Submission Date']
        review_deadline = row['Deadline']
        review_id = row['Review ID']

        if 'Effort' in row:
            effort = row['Effort']

        # Don't include unfinished submissions
        if np.isnan(grades).any():
            continue

        # Don't include explicitly filtered submissions
        if review_id in excluded_submissions:
            if verbose:
                print('excluding review %d' % review_id)
            continue

        grader_idx = grader_idx_lookup[grader]
        assignment_idx = assignment_idx_lookup[assignment]
        week_nums[assignment_idx] = week

        # Calibration grades: save as true grades
        if grader_lookup[grader] == 'instructor':
            calibration_grades[assignment_idx, :] = grades
            assignment_lookup[assignment] = 'calibration'

        # Otherwise, save as observed grades
        else:
            observed_grades[grader_idx, assignment_idx, :] = grades
            graph[grader_idx, assignment_idx] = 1
            review_dates[grader_idx, assignment_idx] = review_date if (type(review_date) == float and not np.isnan(review_date)) else review_deadline
            review_ids[grader_idx, assignment_idx] = review_id

        # Save effort grade
        if 'Effort' in row:
            effort_grades[grader_idx, assignment_idx] = effort

    # Order timeline for each student
    # Sort by review date; break ties with review IDs
    ordered_graph = np.full(graph.shape, -1)
    for grader_idx in range(num_graders):
        grader_assignments = np.where(graph[grader_idx, :] == 1)
        primary_sort = review_dates[grader_idx, grader_assignments]
        secondary_sort = review_ids[grader_idx, grader_assignments]
        timeline_sorted = np.lexsort((secondary_sort, primary_sort))
        ordered_graph[grader_idx, grader_assignments] = timeline_sorted

    return observed_grades, graph, ordered_graph, calibration_grades, effort_grades, week_nums, grader_lookup, assignment_lookup, df

def load_mta_data_new(fnames, excluded_submissions=[], verbose=True):
    """
    Load data from MTA

    Input:
    - fnames: list of paths to CSV files
    - excluded_submissions: list of review IDs to exclude from dataset

    Output:
    - observed_grades: 3D matrix of grades
    - graph: 2D binary matrix showing which grades are legitimate
    - calibration_grades: 3D matrix of true grades; equal to np.nan for regular assignments, and true values for calibrations
    - effort_grades: 2D matrix of responsibilities; equal to 0 or 1 when graded by a TA, or np.nan if not
    - grader_list: OrderedDict of {student ID: 'student' or 'ta'}
    - submission_list: list of submission IDs
    - df: raw dataframe from concatenating weekly data
    """

    # Load data
    dfs = []
    d = {'True': True, 'False': False}
    for (week_num, fname) in enumerate(fnames):
        df_week = pd.read_csv(fname, dtype={'Student ID': str, 'Reviewer ID': str, 'Submission Date': str, 'Deadline': str})
        df_week['Week'] = week_num+1
        dfs.append(df_week)
    df = pd.concat(dfs)
    # Apply other processing
    df['Reviewer Role'] = df['Reviewer Role'].apply(lambda s: s.strip() if type(s) == str else s)
    df['Student ID'] = df['Student ID'].apply(lambda s: s.strip() if type(s) == str else s)
    df['Reviewer ID'] = df['Reviewer ID'].apply(lambda s: s.strip() if type(s) == str else s)
#     df['Review Flagged']= df['Review Flagged'].map(d)
    # Get student list and assignment list
    # Note: misses any students who graded 0 assignments
    grader_lookup = OrderedDict()
    for role in ['student', 'ta', 'instructor']:
        for student in np.unique(df[df['Reviewer Role'] == role]['Reviewer ID']):
            grader_lookup[student] = role
    
    assignment_lookup = OrderedDict()
    for assignment in np.unique(df['Submission ID']):
        assignment_lookup[assignment] = 'submission' # default value

    # Build reverse lookups for speed
    grader_idx_lookup = {student: idx for (idx, student) in enumerate(grader_lookup.keys())}
    assignment_idx_lookup = {assignment: idx for (idx, assignment) in enumerate(assignment_lookup.keys())}

    # Build matrices
    component_names = ['1. Argument structure', '4. English', '2. Evidence', '3. Subject matter']
    num_graders = len(grader_lookup)
    num_assignments = len(assignment_lookup)
    num_components = len(component_names)

    observed_grades = np.zeros((num_graders, num_assignments, num_components))
    calibration_grades = np.full((num_assignments, num_components), np.nan)
    effort_grades = np.full((num_graders, num_assignments), np.nan)
    graph = np.zeros((num_graders, num_assignments))
    review_dates = np.empty((num_graders, num_assignments), dtype=object)
    review_ids = np.zeros((num_graders, num_assignments))
    week_nums = np.zeros(num_assignments)
    flagged_reviews = []
    for index, row in df.iterrows():
#         print(row['Review Flagged'])
        grader = row['Reviewer ID']
        assignment = row['Submission ID']
        week = row['Week']
        grades = [row[component] for component in component_names]
        review_date = row['Submission Date']
        review_deadline = row['Deadline']
        review_id = row['Review ID']

        if 'Effort' in row:
            effort = row['Effort']

        # Don't include unfinished submissions
        if np.isnan(grades).any():
            continue

        # Don't include explicitly filtered submissions
        
        if row['Review Flagged'] == True:
            flagged_reviews.append(row['Review ID'])
        else:
            if review_id in excluded_submissions:
                if verbose:
                    print('excluding review %d' % review_id)
                continue

            grader_idx = grader_idx_lookup[grader]
            assignment_idx = assignment_idx_lookup[assignment]
            week_nums[assignment_idx] = week

            # Calibration grades: save as true grades
            if grader_lookup[grader] == 'instructor':
                calibration_grades[assignment_idx, :] = grades
                assignment_lookup[assignment] = 'calibration'

            # Otherwise, save as observed grades
            else:
                observed_grades[grader_idx, assignment_idx, :] = grades
                graph[grader_idx, assignment_idx] = 1
                review_dates[grader_idx, assignment_idx] = review_date if (type(review_date) == float and not np.isnan(review_date)) else review_deadline
                review_ids[grader_idx, assignment_idx] = review_id

            # Save effort grade
            if 'Effort' in row:
                effort_grades[grader_idx, assignment_idx] = effort

    # Order timeline for each student
    # Sort by review date; break ties with review IDs
    ordered_graph = np.full(graph.shape, -1)
    for grader_idx in range(num_graders):
        grader_assignments = np.where(graph[grader_idx, :] == 1)
        primary_sort = review_dates[grader_idx, grader_assignments]
        secondary_sort = review_ids[grader_idx, grader_assignments]
        timeline_sorted = np.lexsort((secondary_sort, primary_sort))
        ordered_graph[grader_idx, grader_assignments] = timeline_sorted

    return observed_grades, graph, ordered_graph, calibration_grades, effort_grades, week_nums, grader_lookup, assignment_lookup, flagged_reviews, df

def add_calibrations(grades, graph, calibration_grades, calibration_mask, effort_grades=None):
    """
    Update an observed/reported grade matrix and graph to include an "instructor" grader.
    
    Inputs:
    - grades: (graders, assignments, components) matrix of observed or reported grades
    - graph: (graders, assignments) binary matrix
    - calibration_grades: (assignments, components) matrix of instructor-assigned true grades
    - calibration_mask: (assignments) binary vector 
    """
    
    new_grades = np.vstack([grades, [calibration_grades]])
    new_graph = np.vstack([graph, [calibration_mask]])
    if effort_grades is None:
        return (new_grades, new_graph)
    else:
        instructor_effort_grades = np.full((1,effort_grades.shape[1]), np.NaN)
        effort_grades_instructor = np.vstack([effort_grades, instructor_effort_grades])
        return (new_grades, new_graph, effort_grades_instructor)

# TODO: refactor notebooks...
load_fall19_data = load_mta_data


def load_mta_data_from_redis(input_db, excluded_submissions=[], verbose=True, passkey=''):
    db = Database(host=input_db, port=6379, password=passkey)
    lock = db.lock('inference-lock')
    lock.acquire()
    list_of_submissions_with_reviews = db.List('list_of_submissions_with_reviews')
    # Load data
    df = []
    for r in list_of_submissions_with_reviews:
        df.append(db.Hash(r).as_dict(decode=True))
    d = {'True': True, 'False': False}
    df = pd.DataFrame(df).astype({'Student ID':int})
    df = pd.DataFrame(df).astype({'Submission ID':int})
    df = pd.DataFrame(df).astype({'Calibration ID':int})
    df = pd.DataFrame(df).astype({'Review ID':int})
    #df = pd.DataFrame(df).astype({'Reviewer ID':int})
    df = pd.DataFrame(df).astype({'Reviewer Grade':float})
    df = pd.DataFrame(df).astype({'Reviewer Weight':float})
    df['Submission Appealed']= df['Submission Appealed'].map(d)
    df['Review Reported']= df['Review Reported'].map(d)
    df['Review Flagged']= df['Review Flagged'].map(d)
    df = pd.DataFrame(df).astype({'1. Argument structure':float})
    df = pd.DataFrame(df).astype({'4. English':float})
    df = pd.DataFrame(df).astype({'2. Evidence':float})
    df = pd.DataFrame(df).astype({'3. Subject matter':float})
    df = pd.DataFrame(df).astype({'Week':int})
    # for (week_num, fname) in enumerate(fnames):
    #     df_week = pd.read_csv(fname, dtype={'Student ID': str, 'Reviewer ID': str, 'Submission Date': str, 'Deadline': str})
    #     df_week['Week'] = week_num+1
    #     dfs.append(df_week)
    # df = pd.concat(dfs)
    # Apply other processing
    df['Reviewer Role'] = df['Reviewer Role'].apply(lambda s: s.strip() if type(s) == str else s)
    df['Student ID'] = df['Student ID'].apply(lambda s: s.strip() if type(s) == str else s)
    df['Reviewer ID'] = df['Reviewer ID'].apply(lambda s: s.strip() if type(s) == str else s)
    df['Submission Appealed'] = df['Submission Appealed'].apply(lambda s: s.strip() if type(s) == str else s)
    # Get student list and assignment list
    # Note: misses any students who graded 0 assignments
    grader_lookup = OrderedDict()
    for role in ['student', 'ta', 'instructor']:
        for student in np.unique(df[df['Reviewer Role'] == role]['Reviewer ID']):
            grader_lookup[student] = role
    assignment_lookup = OrderedDict()
    for assignment in np.unique(df['Submission ID']):
        student_id = df[df['Submission ID'] == assignment]['Student ID'].iloc[0]
        if str(student_id) in np.unique(df['Reviewer ID']):
            assignment_lookup[assignment] = ['submission', student_id ] # default value
        else: 
            assignment_lookup[assignment] = ['submission', -1 ]
    # Build reverse lookups for speed
    grader_idx_lookup = {student: idx for (idx, student) in enumerate(grader_lookup.keys())}
    assignment_idx_lookup = {assignment: idx for (idx, assignment) in enumerate(assignment_lookup.keys())}
    # Build matrices
    component_names = ['1. Argument structure', '4. English', '2. Evidence', '3. Subject matter']
    num_graders = len(grader_lookup)
    num_assignments = len(assignment_lookup)
    num_components = len(component_names)
    observed_grades = np.zeros((num_graders, num_assignments, num_components))
    calibration_grades = np.full((num_assignments, num_components), np.nan)
    effort_grades = np.full((num_graders, num_assignments), np.nan)
    graph = np.zeros((num_graders, num_assignments))
    review_dates = np.empty((num_graders, num_assignments), dtype=object)
    review_ids = np.zeros((num_graders, num_assignments))
    week_nums = np.zeros(num_assignments)
    for index, row in df.iterrows():
        grader = row['Reviewer ID']
        assignment = row['Submission ID']
        week = row['Week']
        grades = [row[component] for component in component_names]
        review_date = row['Submission Date']
        review_deadline = row['Deadline']
        review_id = row['Review ID']
        if row['Calibration ID'] > 0 and row['Review Flagged'] == True:
            if verbose:
                print('excluding review %d because if was a flagged calibration review' % review_id)
            continue
        if 'Effort' in row:
            effort = row['Effort']
        # Don't include unfinished submissions
        if np.isnan(grades).any():
            continue
        # Don't include explicitly filtered submissions
        if review_id in excluded_submissions:
            if verbose:
                print('excluding review %d' % review_id)
            continue
        grader_idx = grader_idx_lookup[grader]
        assignment_idx = assignment_idx_lookup[assignment]
        week_nums[assignment_idx] = week
        # Calibration grades: save as true grades
        if grader_lookup[grader] == 'instructor':
            calibration_grades[assignment_idx, :] = grades
            assignment_lookup[assignment] = ['calibration' , -1]
        # Otherwise, save as observed grades
        else:
            observed_grades[grader_idx, assignment_idx, :] = grades
            graph[grader_idx, assignment_idx] = 1
            review_dates[grader_idx, assignment_idx] = review_date if (type(review_date) == float and not np.isnan(review_date)) else review_deadline
            review_ids[grader_idx, assignment_idx] = review_id
        # Save effort grade
        if 'Effort' in row:
            effort_grades[grader_idx, assignment_idx] = effort
        if row['Review Flagged'] == True:
            effort_grades[grader_idx, assignment_idx]= 0 
    # Order timeline for each student
    # Sort by review date; break ties with review IDs
    ordered_graph = np.full(graph.shape, -1)
    for grader_idx in range(num_graders):
        grader_assignments = np.where(graph[grader_idx, :] == 1)
        primary_sort = review_dates[grader_idx, grader_assignments]
        secondary_sort = review_ids[grader_idx, grader_assignments]
        timeline_sorted = np.lexsort((secondary_sort, primary_sort))
        ordered_graph[grader_idx, grader_assignments] = timeline_sorted
    lock.release()
    return observed_grades, graph, ordered_graph, calibration_grades, effort_grades, week_nums, grader_lookup, assignment_lookup, df
    # return observed_grades, graph, ordered_graph, calibration_grades, effort_grades, grader_lookup, assignment_lookup, df
