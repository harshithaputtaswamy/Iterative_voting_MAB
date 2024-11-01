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

