
import numpy as np

from matchings import get_cyclic_matching

# Generate data per week
def generate_data(hyperparams, num_students, num_assignments, grades_per_week, effort_p, reliabilities):

    # unpack hyperparams
    (mu_s_dg, sigma_s_dg, a_beta_dg, b_beta_dg, alpha_rel_dg, beta_rel_dg, tau_0_dg) = hyperparams
    
    # Generate true grades
    true_grades = np.random.normal(mu_s_dg, sigma_s_dg, size=num_assignments)

    efforts_vu = np.zeros((num_students, num_assignments))
    efforts_vu[:] = np.nan

    # Generate matching graph
    matchings = get_cyclic_matching(num_students, grades_per_week)
    graph = np.zeros((num_students, num_assignments))
    
    # Generate true grades 
    Z = np.zeros((num_students, num_assignments))
    for (student, assgn) in matchings:
        effort_this_assgn = np.random.binomial(1, p=effort_p[student])
        efforts_vu[student, assgn] = effort_this_assgn
        mu = mu_s_dg
        tau = tau_0_dg
        if effort_this_assgn:
            mu = true_grades[assgn]
            tau = reliabilities[student]
        Z[student, assgn] = np.random.normal(mu,
                                             np.sqrt(1. / tau))
        graph[student, assgn] = 1

    # Round observed grades
    # TODO: make this an option?
    Z = np.rint(Z)

    return Z, graph, true_grades, efforts_vu

# Genearate data for several weeks
def generate_weekly_data(hyperparams, num_students, num_assignments, grades_per_week, num_weeks):
    (mu_s_dg, sigma_s_dg, a_beta_dg, b_beta_dg, alpha_rel_dg, beta_rel_dg, tau_0_dg) = hyperparams
    week_names = []
    Z_all_weeks=[]
    true_grades_all_weeks=[]
    efforts_vu_all_weeks=[]
    graph_all_weeks=[]
    effort_p = np.random.beta(a_beta_dg, b_beta_dg, size=num_students)
    reliabilities = np.random.gamma(alpha_rel_dg, 1./beta_rel_dg,size=num_students)

    for week in range(num_weeks):
        Z, graph, true_grades, efforts_vu = generate_data(
            hyperparams, num_students, num_assignments, grades_per_week , effort_p, reliabilities
        )
        
        week_names.append(['week_' + str(week+1)] * num_assignments)
        Z_all_weeks.append(Z)
        true_grades_all_weeks.append(true_grades)
        efforts_vu_all_weeks.append(efforts_vu)
        graph_all_weeks.append(graph)

    week_names = np.concatenate(week_names)
    Z_all_weeks = np.concatenate(Z_all_weeks, axis=1)
    true_grades_all_weeks = np.concatenate(true_grades_all_weeks, axis=0)
    efforts_vu_all_weeks = np.concatenate(efforts_vu_all_weeks, axis=1)
    graph_all_weeks = np.concatenate(graph_all_weeks, axis=1)

    return week_names, Z_all_weeks, graph_all_weeks, true_grades_all_weeks, effort_p, reliabilities, efforts_vu_all_weeks