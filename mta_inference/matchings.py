import numpy as np
import pandas as pd

def get_cyclic_matching(num_students, grades_per_week):
    """
    Create a "cyclic" matching.
    Students are randomly arranged in a circle, and each student grades a few students in front of them.
    
    Inputs:
    - num_students: number of graders. Assumes that each student also has an assignment to be graded.
    - grades_per_week: number of assignments that each grader will grade (and number of graders per assignment).

    Outputs:
    - List of (grader, assignment) pairs
    """
    matchings = []
    line_up = np.random.permutation(range(num_students))
    for idx, student in enumerate(line_up):
        curr = idx + 1
        next_k = []
        for i in range(grades_per_week):
            next_k.append(line_up[curr % num_students])
            curr += 1
        matchings += [(student, assign) for assign in next_k]
    return matchings

def try_sample_random_matching(num_students, grades_per_week):
    """
    Subroutine for random matching.

    Inputs:
    - num_students: number of graders. Assumes that each student also has an assignment to be graded.
    - grades_per_week: number of assignments that each grader will grade (and number of graders per assignment).

    Outputs:
    - List of (grader, assignment) pairs, or None if matching was invalid 
      (ex: student would have to grade same assignment twice)
    """

    matching = []
    
    # Track how many more assignments each grader can grade
    grades_left = np.array([grades_per_week for _ in range(num_students)])
    
    for student in range(num_students):
        # Find graders that have more capacity (and aren't this student) 
        possible_students = np.where(grades_left > 0)[0].tolist()
        try:
            possible_students.remove(student)
        except ValueError:
            # student wasn't available to grade themselves - not a problem
            pass
        
        # If not enough graders left, we failed
        if len(possible_students) < grades_per_week:
            return None
        
        # Otherwise, add these assignments to the matching
        graders = np.random.choice(possible_students, grades_per_week, replace=False)
        for grader in graders:
            matching.append((grader, student))
            grades_left[grader] -= 1
    
    return matching

def get_random_matching(num_students, grades_per_week):
    """
    Get a random matching.
    Finds a random directed graph where each node has in-degree = out-degree = grades_per_week.

    Inputs:
    - num_students: number of graders. Assumes that each student also has an assignment to be graded.
    - grades_per_week: number of assignments that each grader will grade (and number of graders per assignment).

    Outputs:
    - List of (grader, assignment) pairs
    - Number of attempts required to sample a valid matching
    """        

    # Run rejection sampling until successful
    attempt = 1
    while True:
        matching = try_sample_random_matching(num_students, grades_per_week)
        if matching is not None:
            return (matching, attempt)
        attempt += 1

def try_sample_unequal_matching(graders, assignments, grades_per_week):
    """
    Subroutine for unequal random matching.

    Outputs:
    - List of (grader, assignment) pairs, or None if matching was invalid 
      (ex: student would have to grade same assignment twice)
    """
    
    num_graders = len(graders)
    num_assignments = len(assignments)

    # Find how many grades we can guarantee for each assignment
    #num_total_grades = num_graders * grades_per_week
    num_total_grades = sum(grades_per_week.values())
    min_grades_per_assignment = num_total_grades // num_assignments # rounds down

    # Track how many grades each assignment has so far
    grades_per_assignment = {assignment: 0 for assignment in assignments}

    matching = []
    for grader in graders:
        grader_assignments = []

#        for i in range(grades_per_week):
        for i in range(grades_per_week[grader]):
            # Only allowed to grade an extra time after every assignment has enough grades
            if min(grades_per_assignment.values()) >= min_grades_per_assignment:
                gradeable_assignments = [a for a in assignments if grades_per_assignment[a] < min_grades_per_assignment+1]
            else:
                gradeable_assignments = [a for a in assignments if grades_per_assignment[a] < min_grades_per_assignment]

            # Not allowed to grade self or double-grade any assignments
            for bad_assignment in grader_assignments + [grader]:
                if bad_assignment in gradeable_assignments:
                    gradeable_assignments.remove(bad_assignment)

            # If nothing left to grade, we've failed!
            if len(gradeable_assignments) == 0:
                return None

            # Otherwise, add a random one to the list
            next_assignment = np.random.choice(gradeable_assignments)
            grader_assignments.append(next_assignment)
            grades_per_assignment[next_assignment] += 1

        # Update matching with this grader's assignments
        for assignment in grader_assignments:
            matching.append((grader, assignment))

    return matching

def sample_random_unequal_matching(graders, assignments, grades_per_week):
    """
    Get a random matching without assuming an equal number of assignments and graders.
    Finds a random directed graph where each node has out-degree = grades_per_week and in-degrees are as balanced as possible.

    Inputs:
    - graders: list of grader IDs
    - assignments: list of assignment IDs (or IDs of students that submitted them)
    - grades_per_week: number of assignments that each grader will grade.

    Outputs:
    - List of (grader, assignment) pairs
    - Number of attempts required to sample a valid matching
    """        

    # Run rejection sampling until successful
    attempt = 1
    while True:
        matching = try_sample_unequal_matching(graders, assignments, grades_per_week)
        if matching is not None:
            return (matching, attempt)
        attempt += 1
        
def sample_random_unequal_matching_fast(graders, assignments, grades_per_week):
    """
    Faster matching method.
    Caveats: 
    - len(graders) must be a multiple of len(assignments)
    - Doesn't attempt to avoid assigning a student to their own assignment
    """
    
    num_graders = len(graders)
    num_assignments = len(assignments)
    
    if num_graders % num_assignments != 0:
        raise ValueError("Expected graders to be a multiple of assignments, but got %d graders and %d assignments" % (num_graders, num_assignments))
        
    graders_shuffled = np.array(graders)
    np.random.shuffle(graders_shuffled)
    matching = []
    for class_split in range(num_graders // num_assignments):
        subset_graders = graders_shuffled[class_split*num_assignments:(class_split+1)*num_assignments]
        
        assignments_shuffled = np.array(assignments)
        np.random.shuffle(assignments_shuffled)
        
        for i in range(num_assignments):
            for j in range(grades_per_week):
                matching.append((subset_graders[i], assignments_shuffled[(i+j)%num_assignments]))
    return matching
        

def match_class(fname_cis, fname_submissions, fname_output, calibration_threshold, grades_per_week):
    """
    Match an entire class.

    Input:
    - fname_cis: path to file with student confidence intervals and qualification status
    - fname_submissions: path to file with this week's ungraded assignments
    - fname_output: path to save matching results
    - calibration_threshold: minimum dependability lower bound to get into independent pool
    - grades_per_week: number of assignments for each student to grade

    Output:
    - Dataframe with resulting matching
    - List of (grader, submitter) matching
    """
    # Load CSVs
    student_cis = pd.read_csv(fname_cis, dtype={'Student ID': 'str'})
    submissions = pd.read_csv(fname_submissions, dtype={'Student ID': 'str'})

    # Filter out submissions that are graded
    # TODO: do this properly... for now, just remove calibrations
    submissions = submissions[submissions['Calibration ID'] == 0]
    
    # Split into pools
    independent_students = {}
    supervised_students = {}
    for (idx, row) in student_cis.iterrows():
        if row['Lower Confidence Bound'] < calibration_threshold:
            supervised_students[row['Student ID']] = row['Qualification Status']
        else:
            independent_students[row['Student ID']] = row['Qualification Status']
            
    # Build student -> submission lookup
    submission_lookup = {}
    for (idx, row) in submissions.iterrows():
        submission_lookup[row['Student ID']] = row['Submission ID']
        
        
    # Match each pool
    submitter_matching = []
    for pool in [independent_students, supervised_students]:
        graders = []
        submitters = []
        grades_per_week_pool={}

        for student in pool:
            # If qualified, add to graders
            if pool[student]:
                graders.append(student)
                grades_per_week_pool[student]= grades_per_week[student]
            # If has assignment, add to assignments
            if student in submission_lookup:
                submitters.append(student)
#         print(grades_per_week_pool)
        (pool_matching, attempts) = sample_random_unequal_matching(graders, submitters, grades_per_week_pool)
        print('Matched pool (%d graders, %d assignments) in %d attempts' % (len(graders), len(submitters), attempts))

        submitter_matching += pool_matching

    # Convert submitters to submissions
    matching = [(grader, submission_lookup[submitter]) for (grader, submitter) in submitter_matching]
        
    # Make output and save
    [grader_list, submission_list] = [list(t) for t in zip(*matching)]
    df_output = pd.DataFrame({'submission id': submission_list, 'student id': grader_list})
    df_output.to_csv(fname_output, index=False)

    return df_output, submitter_matching


def match_class_new(independent_students, supervised_students, fname_submissions, fname_output, grades_per_week):
    """
    Match an entire class.

    Input:
    - fname_cis: path to file with student confidence intervals and qualification status
    - fname_submissions: path to file with this week's ungraded assignments
    - fname_output: path to save matching results
    - calibration_threshold: minimum dependability lower bound to get into independent pool
    - grades_per_week: number of assignments for each student to grade

    Output:
    - Dataframe with resulting matching
    - List of (grader, submitter) matching
    """
    # Load CSVs
#     student_cis = pd.read_csv(fname_cis, dtype={'Student ID': 'str'})
    submissions = pd.read_csv(fname_submissions, dtype={'Student ID': 'str'})

    # Filter out submissions that are graded
    # TODO: do this properly... for now, just remove calibrations
    submissions = submissions[submissions['Calibration ID'] == 0]
    
    # Split into pools
#     independent_students = {}
#     supervised_students = {}
#     for (idx, row) in student_cis.iterrows():
#         if row['Lower Confidence Bound'] < calibration_threshold:
#             supervised_students[row['Student ID']] = row['Qualification Status']
#         else:
#             independent_students[row['Student ID']] = row['Qualification Status']
            
    # Build student -> submission lookup
    submission_lookup = {}
    for (idx, row) in submissions.iterrows():
        submission_lookup[row['Student ID']] = row['Submission ID']
        
        
    # Match each pool
    submitter_matching = []
    for pool in [independent_students, supervised_students]:
        graders = []
        submitters = []
        grades_per_week_pool={}

        for student in pool:
            # If qualified, add to graders
            if pool[student]:
                graders.append(student)
                grades_per_week_pool[student]= grades_per_week[student]
            # If has assignment, add to assignments
            if student in submission_lookup:
                submitters.append(student)
#         print(grades_per_week_pool)
        (pool_matching, attempts) = sample_random_unequal_matching(graders, submitters, grades_per_week_pool)
        print('Matched pool (%d graders, %d assignments) in %d attempts' % (len(graders), len(submitters), attempts))

        submitter_matching += pool_matching

    # Convert submitters to submissions
    matching = [(grader, submission_lookup[submitter]) for (grader, submitter) in submitter_matching]
        
    # Make output and save
    [grader_list, submission_list] = [list(t) for t in zip(*matching)]
    df_output = pd.DataFrame({'submission id': submission_list, 'student id': grader_list})
    df_output.to_csv(fname_output, index=False)

    return df_output, submitter_matching