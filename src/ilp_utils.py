import pulp

def chamberlin_courant_borda_utility(ranked_pref, k):
    """
    Solves the Chamberlin-Courant rule using Borda utility with ILP.
    Parameters:
    ranked_pref (dict of list of ints): A matrix where entry [v][i] represents
                                         the rank of candidate in ranked preferences of voters.
    k (int): Committee size.
    Returns:
    list: Selected committee.
    """
    num_voters = len(ranked_pref)
    num_candidates = len(ranked_pref[0])

    # Create the ILP problem
    prob = pulp.LpProblem("Chamberlin-Courant-Borda", pulp.LpMaximize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", range(num_candidates), cat='Binary')  # Candidate selection
    r = pulp.LpVariable.dicts("r", (range(num_voters), range(num_candidates)), cat='Binary')  # Voter assignment

    # Constraint: only k candidates are selected
    prob += pulp.lpSum(x[i] for i in range(num_candidates)) == k

    # Constraint: each voter must be assigned to exactly one candidate
    for v in range(num_voters):
        prob += pulp.lpSum(r[v][i] for i in range(num_candidates)) == 1

    # Constraint: a voter can only be assigned to a candidate if that candidate is selected
    for v in range(num_voters):
        for i in range(num_candidates):
            prob += r[v][i] <= x[i]

    # Objective function: maximize the total Borda score based on assignment
    prob += pulp.lpSum((num_candidates - ranked_pref[v].index(i) - 1) * r[v][i] for v in range(num_voters) for i in range(num_candidates))
    prob.solve(pulp.GUROBI(msg=False))
    selected_candidates = [i for i in range(num_candidates) if x[i].varValue == 1]
    return selected_candidates


def monroe_borda_utility(ranked_pref, k):
    """
    Solves the Monroe rule using Borda utility with ILP.
    
    Parameters:
    ranked_pref (dict of list of ints): A matrix where entry [v][i] represents
                                         the rank of candidate in ranked preferences of voters.
    k (int): Committee size.

    Returns:
    list: Selected committee.
    """

    num_voters = len(ranked_pref)
    num_candidates = len(ranked_pref[0])

    # Create the ILP problem
    prob = pulp.LpProblem("Monroe-Borda", pulp.LpMaximize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", range(num_candidates), cat='Binary')  # Candidate selection
    r = pulp.LpVariable.dicts("r", (range(num_voters), range(num_candidates)), cat='Binary')  # Voter assignment

    # Constraint: only k candidates are selected
    prob += pulp.lpSum(x[i] for i in range(num_candidates)) == k

    # Constraint: each voter must be assigned to exactly one candidate
    for v in range(num_voters):
        prob += pulp.lpSum(r[v][i] for i in range(num_candidates)) == 1

    # Constraint: a voter can only be assigned to a candidate if that candidate is selected
    for v in range(num_voters):
        for i in range(num_candidates):
            prob += r[v][i] <= x[i]

    # Constraint: balaced assignment
    for cand in range(num_candidates):
        prob += pulp.lpSum(r[i][cand] for i in range(num_voters)) <= ((num_voters + k - 1) // k ) * x[cand]
        prob += pulp.lpSum(r[i][cand] for i in range(num_voters)) >= (num_voters // k) * x[cand]

    # Objective function: maximize the total Borda score based on assignment
    prob += pulp.lpSum((num_candidates - ranked_pref[v].index(i) - 1) * r[v][i] for v in range(num_voters) for i in range(num_candidates))
    
    prob.solve(pulp.GUROBI(msg=False))
    selected_candidates = [i for i in range(num_candidates) if x[i].varValue == 1]
    
    return selected_candidates


def pav_utility(ranked_pref, k, approval_dict):
    num_voters = len(ranked_pref)
    num_candidates = len(ranked_pref[0])

    # Create the ILP problem
    prob = pulp.LpProblem("PAV", pulp.LpMaximize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", range(num_candidates), cat='Binary')  # Candidate selection
    s = pulp.LpVariable.dicts("s", (range(num_voters), range(1, k + 1)), lowBound=0, cat="Continuous") # Satisfaction score of voter i
    # TODO: s[i][l]
    # Constraint: only k candidates are selected
    prob += pulp.lpSum(x[i] for i in range(num_candidates)) == k

    for i in range(num_voters):
        # prob += s[i] <= pulp.lpSum(approval_dict[i][j] * x[j] for j in range(num_candidates))
        for l in range(1, k + 1):  # Iterate up to the committee size
            prob += s[i][l] <= (pulp.lpSum(approval_dict[i][j] * x[j] for j in range(num_candidates)) / l)
            prob += s[i][l] >= (pulp.lpSum(approval_dict[i][j] * x[j] for j in range(num_candidates)) - l + 1 / k)

    # Objective function: maximize the total score
    # for l in range(1, k + 1):
    #     prob += pulp.lpSum(s[i] *(1/l) for i in range(num_voters))
    z = pulp.LpVariable.dicts("z", [(i, l) for i in range(num_voters) for l in range(1, k + 1)], 
                              lowBound=0, cat="Continuous")
    for i in range(num_voters):
        for l in range(1, k + 1):
            prob += z[(i, l)] * l == s[i][l]  # z[i] = s[i] / l

    # Objective function: maximize the total score
    prob += pulp.lpSum(z[(i, l)] for i in range(num_voters) for l in range(1, k + 1))

    prob.solve(pulp.GUROBI(msg=False))
    selected_candidates = [i for i in range(num_candidates) if x[i].varValue == 1]
    
    return selected_candidates

