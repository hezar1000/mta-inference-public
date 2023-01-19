import pulp
import numpy as np
import itertools

# load solver on import
# solver = pulp.getSolver('CPLEX_PY', timeLimit=10, msg=True) # note: requires cplex python API
# solver = pulp.getSolver('CPLEX_CMD', timeLimit=2, msg=True, keepFiles=True) 
solver = pulp.getSolver('PULP_CBC_CMD', msg=False) 


def optimize_explanation(reported_grades, posterior_mass, desired_weights, max_weight_change, min_weight, penalty_coeff):
    """
    Find a rounded convex combination of the grades that maximizes the posterior mass.

    Solves:
        max \sum_{j,k} m_{j,k} g_{j,k} - P \sum_i p_i + n_i
    subject to:
        \sum_k g_{j,k} = 1               (g_{j,k} is one-hot vector)
        g_j = \sum_k k g_{j,k}           (g_{j,:} vector represents final grade g_j)
        g_j = \sum_i w_i r_{i,j} + s_j   (grade for component j is rounded weighted average)
        w_i = d_i + (p_i - n_i)          (final weights are close to desired weights)
        w_i >= M c_i                     (weights are 0 or >= M)
        w_i <= c_i                       (weights are 0 or >= M)
        \sum_i w_i = 1                   (weights are convex combination of graders)
    where
        0 <= w_i <= 1 for all graders i
        g_j \in {0, 1, 2, ..., max_grade} for all components j
        g_{j,k} \in {0, 1} for all components j and grades k
        -0.5 <= s_j <= 0.5 for all components j
        0 <= p_i, n_i <= S for all graders i
        c_i \in {0, 1} for all graders i

    Inputs:
    - reported_grades: (num_graders, num_components) array of reported grades from each grader
    - posterior_mass: (num_components, num_possible_grades) array of posterior mass on each bin. sum on second axis = 1.
    - desired_weights: (num_graders) list of desired weights
    - max_weight_change: max allowed difference between desired weights and returned weights
    - min_weight: minimum non-zero weight allowed for any grader
    - penalty_coeff: scale applied to weight change penalty

    Outputs: tuple of
    - final_weights: (num_graders) list of weights in [0, 1]
    - final_grades: (num_components) list of integer-valued grades
    - objective_value: total amount of mass captured by weighted average
    - status: solver status
    """

    num_graders, num_components = reported_grades.shape
    _, num_grades = posterior_mass.shape

    weights = pulp.LpVariable.dicts('weights', range(num_graders), lowBound=0, upBound=1, cat='Continuous')
    weight_slack_positive = pulp.LpVariable.dicts('weight_slack_positive', range(num_graders), lowBound=0, upBound=max_weight_change, cat='Continuous')
    weight_slack_negative = pulp.LpVariable.dicts('weight_slack_negative', range(num_graders), lowBound=0, upBound=max_weight_change, cat='Continuous')
    weights_nonzero = pulp.LpVariable.dicts('weights_nonzero', range(num_graders), cat='Binary')
    grades = pulp.LpVariable.dicts('grades', range(num_components), lowBound=0, upBound=num_grades-1, cat='Integer')
    grade_vectors = pulp.LpVariable.dicts('grade_vectors', itertools.product(range(num_components), range(num_grades)), cat='Binary')
    grade_slack = pulp.LpVariable.dicts('grade_slack', range(num_components), lowBound=-0.5, upBound=0.5, cat='Continuous')

    prob = pulp.LpProblem('explain_grades', sense=pulp.LpMaximize)

    # objective: total posterior mass, minus L1 penalty for changing from desired weights
    prob += (
        pulp.lpSum(
            [
                grade_vectors[ij] * posterior_mass[ij]
                for ij in itertools.product(range(num_components), range(num_grades)) 
            ]
        ) - penalty_coeff * pulp.lpSum(list(weight_slack_positive.values()) + list(weight_slack_negative.values())),
        "maximize_total_mass"
    )

    # constraint: can only choose one grade for each component
    for component in range(num_components):
        prob += (
            pulp.lpSum([grade_vectors[component, i] for i in range(num_grades)]) == 1,
            "one_hot_grade_%d" % component
        )

    # constraint: grade vector is one-hot representation of grades
    for component in range(num_components):
        prob += (
            grades[component] == pulp.lpSum([
                grade * grade_vectors[component, grade] for grade in range(num_grades)
            ]),
            "one_hot_grade_equal_%d" % component 
        )

    # constraint: grades must be rounded weighted average
    for component in range(num_components):
        prob += (
            pulp.lpSum([
                weights[grader] * reported_grades[grader, component] for grader in range(num_graders)
            ]) == grades[component] + grade_slack[component] 
        )

    # constraint: weights must be close to desired weights
    for grader in range(num_graders):
        prob += (
            weights[grader] == desired_weights[grader] + weight_slack_positive[grader] - weight_slack_negative[grader],
            "weight_change_%d" % grader    
        )

    # constraint: weights must either be zero or above minimum allowed value
    for grader in range(num_graders):
        # if weight is non-zero, enforce lower bound
        prob += (
            weights[grader] >= weights_nonzero[grader] * min_weight,
            "min_nonzero_weight_%d" % grader
        )
        # if weight is zero, enforce upper bound
        prob += (
            weights[grader] <= weights_nonzero[grader],
            "max_zero_weight_%d" % grader
        )

    # constraint: weights must sum to 1
    prob += pulp.lpSum(
        weights[i] for i in range(num_graders)
    ) == 1, "total_weight"

    # prob.writeLP('../data/explain_grades.lp')
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    if prob.status != pulp.constants.LpStatusOptimal:
        raise ValueError("Couldn't find optimal solution. PuLP status: %s" % status)

    objective_value = pulp.value(prob.objective)

    final_weights = np.array([weights[i].varValue for i in range(num_graders)])
    final_grades = np.array([grades[i].varValue for i in range(num_components)])

    # debugging
    # final_weights_nonzero = np.array([weights_nonzero[i].varValue for i in range(num_graders)])
    # print(final_weights_nonzero)
    # print(np.array([weight_slack_positive[i].varValue for i in range(num_graders)]))
    # print(np.array([weight_slack_negative[i].varValue for i in range(num_graders)]))

    return final_weights, final_grades, objective_value, status


def optimize_explanation_old(reported_grades, posterior_mass, trust, trust_const, min_weight):
    """
    Find a rounded convex combination of the grades that maximizes the posterior mass.

    Inputs:
    - reported_grades: (num_graders, num_components) array of reported grades from each grader
    - posterior_mass: (num_components, num_possible_grades) array of posterior mass on each bin. sum on second axis = 1.
    - trust: (num_graders) list of trust levels
    - trust_const: real-valued gap in weights between each trust level
    - min_weight: minimum non-zero weight allowed for any grader

    Outputs: tuple of
    - final_weights: (num_graders) list of weights in [0, 1]
    - final_grades: (num_components) list of integer-valued grades
    - objective_value: total amount of mass captured by weighted average
    - status: solver status
    """

    num_graders, num_components = reported_grades.shape
    _, num_grades = posterior_mass.shape

    weights = pulp.LpVariable.dicts('weights', range(num_graders), lowBound=0, upBound=1, cat='Continuous')
    weights_nonzero = pulp.LpVariable.dicts('weights_nonzero', range(num_graders), cat='Binary')
    grades = pulp.LpVariable.dicts('grades', range(num_components), lowBound=0, upBound=num_grades-1, cat='Integer')
    grade_vectors = pulp.LpVariable.dicts('grade_vectors', itertools.product(range(num_components), range(num_grades)), cat='Binary')
    slack = pulp.LpVariable.dicts('slack', range(num_components), lowBound=-0.5, upBound=0.5, cat='Continuous')

    prob = pulp.LpProblem('explain_grades', sense=pulp.LpMaximize)

    # objective: total posterior mass
    prob += (
        pulp.lpSum(
            [
                grade_vectors[ij] * posterior_mass[ij]
                for ij in itertools.product(range(num_components), range(num_grades)) 
            ]
        ),
        "maximize_total_mass"
    )

    # constraint: can only choose one grade for each component
    for component in range(num_components):
        prob += (
            pulp.lpSum([grade_vectors[component, i] for i in range(num_grades)]) == 1,
            "one_hot_grade_%d" % component
        )

    # constraint: grade vector is one-hot representation of grades
    for component in range(num_components):
        for grade in range(num_grades):
            prob += (
                grades[component] <= grade + 10 * (1 - grade_vectors[component, grade]),
                "grade_ub_%d_%d" % (component, grade)
            )  
            prob += (
                grades[component] >= grade - 10 * (1 - grade_vectors[component, grade]),
                "grade_lb_%d_%d" % (component, grade)
            )  

    # constraint: grades must be rounded weighted average
    for component in range(num_components):
        prob += (
            pulp.lpSum([
                weights[grader] * reported_grades[grader, component] for grader in range(num_graders)
            ]) == grades[component] + slack[component] 
        )

    # constraint: must put higher weight on more trustworthy graders
    for grader_i in range(num_graders):
        for grader_j in range(num_graders):
            if trust[grader_i] > trust[grader_j]:
                trust_gap = trust[grader_i] - trust[grader_j]
                prob += (
                    weights[grader_i] >= weights[grader_j] + trust_const * trust_gap,
                    "trust_gap_%d_%d" % (grader_i, grader_j)
                )

    # constraint: weights must either be zero or above minimum allowed value
    for grader in range(num_graders):
        # if weight is non-zero, enforce lower bound
        prob += (
            weights[grader] >= weights_nonzero[grader] * min_weight,
            "min_nonzero_weight_%d" % grader
        )
        # if weight is zero, enforce upper bound
        prob += (
            weights[grader] <= weights_nonzero[grader],
            "max_zero_weight_%d" % grader
        )

    # constraint: weights must sum to 1
    prob += pulp.lpSum(
        weights[i] for i in range(num_graders)
    ) == 1, "total_weight"

    # TODO: add verbose 
    # prob.writeLP('explain_grades.lp')
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    # print('Status:', pulp.LpStatus[prob.status])
    if prob.status != pulp.constants.LpStatusOptimal:
        raise ValueError("Couldn't find optimal solution. PuLP status: %s" % status)

    objective_value = pulp.value(prob.objective)

    final_weights = np.array([weights[i].varValue for i in range(num_graders)])
    final_grades = np.array([grades[i].varValue for i in range(num_components)])
    final_weights_nonzero = np.array([weights_nonzero[i].varValue for i in range(num_graders)])
    print(final_weights_nonzero)

    return final_weights, final_grades, objective_value, status

if __name__ == "__main__":
    # test on example
    reported_grades = np.array([
        [3, 3, 4, 4],
        [3, 4, 5, 5],
        [3, 4, 4, 5],
        [4, 3, 4, 4],
        [4, 4, 5, 5],
    ])

    posterior_mass = np.array([
        [0.0, 0.1, 0.2, 0.5, 0.2, 0.0],
        [0.0, 0.1, 0.1, 0.2, 0.4, 0.2],
        [0.0, 0.0, 0.1, 0.1, 0.5, 0.3],
        [0.0, 0.0, 0.0, 0.1, 0.4, 0.5],
    ])

    dependabilities = np.array([2, 0, 0, 1, 1])

    final_weights, final_grades, total_mass, status = optimize_explanation(
        reported_grades, 
        posterior_mass, 
        dependabilities / dependabilities.sum(), 
        max_weight_change = 0.2,
        min_weight = 0.1,
        penalty_coeff = 1e-2
    )
    
    print(final_weights)
    print(final_grades)
    print(total_mass)
    print(status)

    # Feasible example
    # final_weights, final_grades, total_mass, status = optimize_explanation_old(
    #     reported_grades, 
    #     posterior_mass, 
    #     trust, 
    #     trust_const = 0.05,
    #     min_weight = 0.2
    # )
    # print(final_weights)
    # print(final_grades)
    # print(total_mass)
    # print(status)

    # # Infeasible example
    # final_weights, final_grades, total_mass, status = optimize_explanation(
    #     reported_grades, 
    #     posterior_mass, 
    #     trust, 
    #     trust_const = 0.05,
    #     min_weight = 0.8
    # )
