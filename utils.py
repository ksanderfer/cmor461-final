import pulp
import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog


def generate_corr_matrix_factor(n, n_factors=2, noise_scale=0.5, rng=None):
    """
    Factor-model correlation matrix with heterogeneous off-diagonals.

    Sigma = L L^T + (1/noise_scale) * I,  L ~ N(0, 1) of shape (n, n_factors),
    then normalized to unit diagonal. Fewer factors / smaller noise_scale ->
    higher average correlation; larger noise_scale -> closer to independent.
    """
    if rng is None:
        rng = np.random.default_rng()
    L = rng.normal(0, 1, size=(n, n_factors))
    Sigma = L @ L.T + np.eye(n) * (1 / noise_scale)
    d = np.sqrt(np.diag(Sigma))
    return Sigma / np.outer(d, d)


def corr_summary(M):
    """Return (mean_abs_off, std_off, max_off) for the off-diagonal entries."""
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = M[mask]
    return float(np.abs(off).mean()), float(off.std()), float(off.max())


def build_A_ub(arcs, n_fac, n_dem):
    """Dense A_ub for the transportation LP. Rows 0..n_fac-1 are capacity
    constraints (one per facility); rows n_fac..n_fac+n_dem-1 are demand
    constraints (one per demand node); columns are arcs in the given order.

    Depends only on the arc list, so build once per configuration and reuse
    across replications.
    """
    n_arcs = len(arcs)
    A = np.zeros((n_fac + n_dem, n_arcs))
    for k, (i, j) in enumerate(arcs):
        A[i, k] = 1.0
        A[n_fac + j, k] = 1.0
    return A


def solve_transportation_fast(arcs, supply, demand, A_ub=None):
    """HiGHS-backed transportation LP. Returns lost sales (unserved demand).

    Same model as solve_transportation but ~100x faster by avoiding CBC
    startup and PuLP object construction per call. Pass in a prebuilt A_ub
    from build_A_ub(arcs, n_fac, n_dem) to reuse it across replications.
    """
    supply = np.asarray(supply, dtype=float)
    demand = np.asarray(demand, dtype=float)
    total_demand = float(demand.sum())
    if len(arcs) == 0 or supply.sum() == 0.0:
        return total_demand
    if A_ub is None:
        A_ub = build_A_ub(arcs, len(supply), len(demand))
    c = -np.ones(len(arcs))
    b_ub = np.concatenate([supply, demand])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    if not res.success:
        return total_demand
    served = -float(res.fun)
    return max(0.0, total_demand - served)


def generate_corr_matrix(n, target_rho, noise_scale=0.1, rng=None):
    """
    Generate a correlation matrix with average off-diagonal correlation
    approximately equal to target_rho.
    
    Uses equicorrelation base + symmetric Gaussian noise + PSD projection.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Equicorrelation base: Sigma = (1 - rho) * I + rho * 1 1^T
    base = (1 - target_rho) * np.eye(n) + target_rho * np.ones((n, n))
    
    # Symmetric noise on off-diagonals
    noise = rng.normal(0, noise_scale, size=(n, n))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0)
    
    M = base + noise
    
    # Project to nearest PSD correlation matrix
    # Eigendecomposition, clip negative eigenvalues, renormalize diagonal
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-6, None)
    M = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Rescale to unit diagonal (this is a correlation matrix, not covariance)
    d = np.sqrt(np.diag(M))
    M = M / np.outer(d, d)
    
    return M

def correlated_bernoulli(p, corr_matrix, n_samples=1):
    L = np.linalg.cholesky(corr_matrix)
    z = L @ np.random.randn(len(corr_matrix), n_samples)
    u = norm.cdf(z)
    result = (u < p).astype(int)
    return result.squeeze() if n_samples == 1 else result


def solve_transportation(arcs, supply, demand):
    prob = pulp.LpProblem("transportation", pulp.LpMaximize)

    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0) for i, j in arcs}

    prob += pulp.lpSum(x.values())

    for i, cap in enumerate(supply):
        prob += pulp.lpSum(x[i, j] for (si, j) in arcs if si == i) <= cap

    for j, dem in enumerate(demand):
        prob += pulp.lpSum(x[i, j] for (i, sj) in arcs if sj == j) <= dem

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    total_flow = pulp.value(prob.objective)
    return sum(demand) - total_flow


def add_smart_links(corr):
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