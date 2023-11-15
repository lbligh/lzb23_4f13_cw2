from scipy.stats import norm
import numpy as np
from typing import Iterable


def eprank_non_iter(game_mat, num_players, num_iters) -> Iterable:
    """

    :param game_mat: Game outcomes
    :param num_players: number of players
    :param num_iters: number of iterations of message passing
    :return: mean and precisions for each players skills based on message passing
    """
    # number of games
    N = game_mat.shape[0]

    # prior skill variance (prior mean is always 0)
    pv = 0.5

    # Helper functions
    psi = lambda x: norm.pdf(x) / norm.cdf(x)
    lam = lambda x: psi(x) * (psi(x) + x)

    # intialize marginal means and precisions
    Ms = np.zeros(num_players)
    Ps = np.zeros(num_players)

    # initialize matrices of game to skill messages, means and precisions
    Mgs = np.zeros((N, 2))
    Pgs = np.zeros((N, 2))
    last_Ms, last_Ps = 0, 1
    # initialize matrices of game to skill to game messages, means and precisions
    Msg = np.zeros((N, 2))
    Psg = np.zeros((N, 2))

    for iter in range(num_iters):
        for p in range(num_players):  # compute marginal player skills
            games_won = np.where(game_mat[:, 0] == p)[0]
            games_lost = np.where(game_mat[:, 1] == p)[0]
            Ps[p] = 1.0 / pv + np.sum(Pgs[games_won, 0]) + np.sum(Pgs[games_lost, 1])
            Ms[p] = (
                np.sum(Pgs[games_won, 0] * Mgs[games_won, 0]) / Ps[p]
                + np.sum(Pgs[games_lost, 1] * Mgs[games_lost, 1]) / Ps[p]
            )

        # (2) compute skill to game messages
        Psg = Ps[game_mat] - Pgs
        Msg = (Ps[game_mat] * Ms[game_mat] - Pgs * Mgs) / Psg

        # (3) compute game to performance messages
        vgt = 1 + np.sum(1.0 / Psg, axis=1)
        mgt = Msg[:, 0] - Msg[:, 1]

        # (4) approximate the marginal on performance differences
        Mt = mgt + np.sqrt(vgt) * psi(mgt / np.sqrt(vgt))
        Pt = 1.0 / (vgt * (1 - lam(mgt / np.sqrt(vgt))))

        # (5) compute performance to game messages
        ptg = Pt - 1.0 / vgt
        mtg = (Mt * Pt - mgt / vgt) / ptg

        # (6) compute game to skills messages
        Pgs = 1.0 / (1 + 1.0 / ptg[:, None] + 1.0 / np.flip(Psg, axis=1))
        Mgs = np.stack([mtg, -mtg], axis=1) + np.flip(Msg, axis=1)

    return Ms, Ps


def eprank_iter(game_mat, num_players, num_iters) -> Iterable:
    """

    :param game_mat: Game outcomes
    :param num_players: number of players
    :param num_iters: number of iterations of message passing
    :return: mean and precisions for each players skills based on message passing
    """
    # number of games
    N = game_mat.shape[0]

    # prior skill variance (prior mean is always 0)
    pv = 0.5

    # Helper functions
    psi = lambda x: norm.pdf(x) / norm.cdf(x)
    lam = lambda x: psi(x) * (psi(x) + x)

    # intialize marginal means and precisions
    Ms = np.zeros(num_players)
    Ps = np.zeros(num_players)

    # initialize matrices of game to skill messages, means and precisions
    Mgs = np.zeros((N, 2))
    Pgs = np.zeros((N, 2))
    last_Ms, last_Ps = 0, 1
    # initialize matrices of game to skill to game messages, means and precisions
    Msg = np.zeros((N, 2))
    Psg = np.zeros((N, 2))

    for iter in range(num_iters):
        for p in range(num_players):  # compute marginal player skills
            games_won = np.where(game_mat[:, 0] == p)[0]
            games_lost = np.where(game_mat[:, 1] == p)[0]
            Ps[p] = 1.0 / pv + np.sum(Pgs[games_won, 0]) + np.sum(Pgs[games_lost, 1])
            Ms[p] = (
                np.sum(Pgs[games_won, 0] * Mgs[games_won, 0]) / Ps[p]
                + np.sum(Pgs[games_lost, 1] * Mgs[games_lost, 1]) / Ps[p]
            )

        # (2) compute skill to game messages
        Psg = Ps[game_mat] - Pgs
        Msg = (Ps[game_mat] * Ms[game_mat] - Pgs * Mgs) / Psg

        # (3) compute game to performance messages
        vgt = 1 + np.sum(1.0 / Psg, axis=1)
        mgt = Msg[:, 0] - Msg[:, 1]

        # (4) approximate the marginal on performance differences
        Mt = mgt + np.sqrt(vgt) * psi(mgt / np.sqrt(vgt))
        Pt = 1.0 / (vgt * (1 - lam(mgt / np.sqrt(vgt))))

        # (5) compute performance to game messages
        ptg = Pt - 1.0 / vgt
        mtg = (Mt * Pt - mgt / vgt) / ptg

        # (6) compute game to skills messages
        Pgs = 1.0 / (1 + 1.0 / ptg[:, None] + 1.0 / np.flip(Psg, axis=1))
        Mgs = np.stack([mtg, -mtg], axis=1) + np.flip(Msg, axis=1)

        yield np.copy(Ms), np.copy(Ps)
