from itertools import combinations
from typing import Sequence

import numpy as np
from numpy import linalg

from pre_processing.classes.CSP import CSP
from pre_processing.classes.CSPApplier import CSPApplier
from pre_processing.classes.ClassTrails import ClassTrails
from pre_processing.reformat_input import reformat_input


def cov(trials):
    """Calculate the covariance for each trial and return their average"""
    n_channels, n_samples, n_trials = trials.shape
    covs = [trials[:, :, i].dot(trials[:, :, i].T) / n_samples for i in range(n_trials)]
    return np.mean(covs, axis=0)


def whitening(sigma):
    """Calculate a whitening matrix for covariance matrix sigma."""
    U, l, _ = linalg.svd(sigma)
    return U.dot(np.diag(l**-0.5))


def gen_csp(class_separated_trails: Sequence[ClassTrails]) -> CSPApplier:
    """
    Calculate the CSP transformation matrix W.
    arguments:
        trials_r - Array (channels x samples x trials) containing right hand movement trials
        trials_f - Array (channels x samples x trials) containing foot movement trials
    returns:
        Mixing matrix W
    """
    csps = []
    for trails_1, trails_2 in combinations(class_separated_trails, 2):
        cov_1 = cov(reformat_input(trails_1.data))
        cov_2 = cov(reformat_input(trails_2.data))
        P = whitening(cov_1 + cov_2)
        B, _, _ = linalg.svd(P.T.dot(cov_2).dot(P))
        csps.append(CSP((trails_1.class_, trails_2.class_), P.dot(B)))
    return CSPApplier(csps)
