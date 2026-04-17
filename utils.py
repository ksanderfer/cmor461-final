import pulp

def add_links(corr):
    '''
    given a correlation matrix, solve an IP as follows:
    min Sum(c_ij)
    s.t.:
        we add a number of links equal to the dimension of the matrix
        the links form a single cycle with no repeated stops
    '''
    n = len(corr)
    prob = pulp.LpProblem("min_correlation_cycle", pulp.LpMinimize)

    # x[i][j] = 1 if link i->j is selected (diagonal forced to 0)
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat='Binary') if i != j else 0
          for j in range(n)] for i in range(n)]

    # MTZ position variables for subtour elimination (node 0 is the fixed anchor)
    u = [pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n - 1, cat='Continuous')
         for i in range(n)]

    prob += pulp.lpSum(corr[i][j] * x[i][j]
                       for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n) if i != j) == 1  # one outgoing
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(n) if i != j) == 1  # one incoming

    # MTZ subtour elimination (excludes node 0 from u constraints)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    links = [(i, j) for i in range(n) for j in range(n)
             if i != j and pulp.value(x[i][j]) > 0.5]
    return links