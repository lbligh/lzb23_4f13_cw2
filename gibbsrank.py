import scipy.linalg
import numpy as np
from tqdm import tqdm


def gibbs_sample(G, num_players, num_iters):
    """Sample Player skills using gibbs sampling

    Takes:
        G - game array: G[i,0] is winner of game i, G[i,1] is loser
        num_players - num of players
        num_iters - obvious

    Returns:
        skills_samples - skills_samples[i,j] is sample for skill of player i at iteration j
    """
    # number of games
    num_games = G.shape[0]
    # Array containing mean skills of each player, set to prior mean (initially set to zero)
    w = np.zeros((num_players, 1))
    # Array that will contain skill samples
    skill_samples = np.zeros((num_players, num_iters))
    # Array containing skill variance for each player, set to prior variance
    pv = 0.5 * np.ones(num_players)
    # number of iterations of Gibbs
    for i in tqdm(range(num_iters)):
        # sample performance given differences in skills and outcomes
        t = np.zeros((num_games, 1))
        for g in range(num_games):
            s = w[G[g, 0]] - w[G[g, 1]]  # skill of winner minus skill of loseer
            t[g] = s + np.random.randn()  # Sample performance (skill dif plus N(0,1))
            while t[g] < 0:  # rejection step
                t[g] = s + np.random.randn()  # resample if rejected

        # Jointly sample skills given performance differences
        m = np.zeros((num_players, 1))
        for p in range(num_players):
            # for each player
            # get sum of all perf differences for won games - sum of all perf differences for lost games
            # prior mean for all players is zero
            m[p] = sum(t[np.where(G[:, 0] == p)]) - sum(
                t[np.where(G[:, 1] == p)]
            )  # LUKE DONE

        iS = np.zeros(
            (num_players, num_players)
        )  # Container for sum of precision matrices (likelihood terms)

        for g in range(num_games):  # for game in games
            winner = G[g, 0]
            loser = G[g, 1]

            iS[winner, winner] += 1
            iS[loser, loser] += 1
            iS[winner, loser] -= 1
            iS[loser, winner] -= 1

        # Posterior precision matrix
        iSS = iS + np.diag(1.0 / pv)

        # Use Cholesky decomposition to sample from a multivariate Gaussian
        iR = scipy.linalg.cho_factor(
            iSS
        )  # Cholesky decomposition of the posterior precision matrix
        mu = scipy.linalg.cho_solve(
            iR, m, check_finite=False
        )  # uses cholesky factor to compute inv(iSS) @ m

        # sample from N(mu, inv(iSS))
        w = mu + scipy.linalg.solve_triangular(
            iR[0], np.random.randn(num_players, 1), check_finite=False
        )
        skill_samples[:, i] = w[:, 0]
    return skill_samples
