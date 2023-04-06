# Import different paths
import numpy as np
import copy
import math
import scipy
import scipy.linalg
from scipy.stats import binom as scp_binom

import scipy.optimize
from functools import partial

# local imports
from .queries import phase_sequence, probability_target


def kl_divergence(P, Q):
    """
    Computes the KL divergence for random variables P and Q
        P: Probability vector of P for different values of range
        Q: Probability vector of P for different values of range
    """
    if len(P) != len(Q):
        raise ValueError("Require prob vectors of P and Q to be of same length!")

    eps = 1e-6
    prob_P = copy.deepcopy(P) + eps
    prob_Q = copy.deepcopy(Q) + eps

    divergence = np.sum(prob_P * np.log(prob_P / prob_Q))
    return divergence


def chernoff_information(gamma, P, Q):
    """
    Computes the chernoff information (Section 11.7, Cover and Thomas) for a given value of gamma

    Output: sum_x P(x)^{gamma} Q(x)^{1 - gamma}

    Inputs:
        P: Probability vector of P for different values of range
        Q: Probability vector of P for different values of range
    """
    if len(P) != len(Q):
        raise ValueError("Require prob vectors of P and Q to be of same length!")

    if gamma > 1 or gamma < 0:
        raise ValueError("Gamma has to be in [0,1]")

    prob_P = copy.deepcopy(P)
    prob_Q = copy.deepcopy(Q)

    C = np.log( np.sum( (prob_P**gamma) * (prob_Q**(1-gamma)) ) )

    return C


def optimize_chernoff_information(prob_P, prob_Q, init_gamma=0.01):
    cost_func = partial(chernoff_information, P=prob_P, Q=prob_Q)
    #res = scipy.optimize.minimize(cost_func, init_gamma, method='trust-constr', jac=None, bounds=[(0, 1)])
    res = scipy.optimize.minimize(cost_func, init_gamma, method='L-BFGS-B', jac=None, bounds=[(0, 1)])

    return res.x


def likelihood_ratio_test(m_shots, outcomes, theta1, prob_theta0, prob_theta1):
    Y1 = int(np.sum(outcomes))
    Y0 = m_shots - Y1

    # Get log-likelihoods
    ll_theta0 = Y1*np.log(prob_theta0[1] + 1e-15) + Y0*np.log(prob_theta0[0] + 1e-15)
    ll_theta1 = Y1 * np.log(prob_theta1[1] + 1e-15) + Y0 * np.log(prob_theta1[0] + 1e-15)

    if ll_theta1 >= ll_theta0:
        theta_est = theta1
    else:
        theta_est = 0.0

    return theta_est


# Function relevant for multi-shot strategy with constant depth
def error_majority_vote(m, angle_vec, theta_1, K, FLAG_math_compute=False):
    """
    Note that we consider y=0 for theta_V=alpha and y=1 for theta_V=0
        m: Number of shots
        angle_vec: phase sequences [psi_W, phi0, ..., phiK]
        theta_1: True value of alpha
        K: Number of "queries"
        FLAG_math_compute: Switch between to (numerically unstable) binomial coeff compute and (stable) cdf compute

    Return: Error from majority error
    """
    if len(angle_vec) != K + 2:
        raise ValueError("Check length of angle vector")

    psi_W = angle_vec[0]
    phi = angle_vec[1:]

    # theta_V = 0
    _, rho = phase_sequence(0.0, phi, psi_W, K)
    prob_theta0 = probability_target(rho)
    p0_theta0, p1_theta0 = prob_theta0

    # theta_V = theta_1
    _, rho = phase_sequence(theta_1, phi, psi_W, K)
    prob_theta1 = probability_target(rho)
    p0_theta1, p1_theta1 = prob_theta1

    # Set limits for computations below
    if np.mod(m, 2) == 0:
        low_m = int(m / 2)
        high_m = int(m / 2) + 1

        # For value of
    else:
        low_m = int(np.floor(m / 2))
        high_m = int(np.ceil(m / 2))

    prob_majority_0_theta0 = 0.0
    prob_majority_1_theta1 = 0.0
    if FLAG_math_compute:
        # Below suffers from math overflow errors
        # See Ref: https://stackoverflow.com/questions/26560726/python-binomial-coefficient
        for ind in range(high_m, m+1):
            prob_majority_0_theta0 += math.comb(m, ind) * ((p0_theta0)**(ind)) * ((p1_theta0)**(m - ind))
            prob_majority_1_theta1 += math.comb(m, ind) * ((p1_theta1)**(ind)) * ((p0_theta1)**(m - ind))

        if np.mod(m, 2) == 0:
            # Add up equal decision for m/2
            ind = int(m/2)
            prob_majority_0_theta0 += 0.5*math.comb(m, ind) * ((p0_theta0) ** (ind)) * ((p1_theta0) ** (m - ind))
            prob_majority_1_theta1 += 0.5*math.comb(m, ind) * ((p1_theta1) ** (ind)) * ((p0_theta1) ** (m - ind))
    else:
        prob_majority_0_theta0 = 1.0 - scp_binom.cdf(low_m, m, p0_theta0)
        prob_majority_1_theta1 = 1.0 - scp_binom.cdf(low_m, m, p1_theta1)

        if np.mod(m, 2) == 0:
            # Add up equal decision for m/2
            prob_majority_0_theta0 += 0.5*scp_binom.pmf(int(m/2), m, p0_theta0)
            prob_majority_1_theta1 += 0.5*scp_binom.pmf(int(m/2), m, p1_theta1)

    error = 0.5*(prob_majority_0_theta0 + prob_majority_1_theta1)

    return error


def error_likelihood_ratio_test(m, angle_vec, theta_1, K, FLAG_kl=True,
                                gamma=0.5, FLAG_optimize_gamma=False, FLAG_verbose=False):
    """
    Note that we consider y=0 for theta_V=alpha and y=1 for theta_V=0
        m: Number of shots
        angle_vec: phase sequences [psi_W, phi0, ..., phiK]
        theta_1: True value of alpha
        K: Number of "queries"
        FLAG_math_compute: Switch between to (numerically unstable) binomial coeff compute and (stable) cdf compute

    Return: Error from majority error
    """
    if len(angle_vec) != K + 2:
        raise ValueError("Check length of angle vector")

    psi_W = angle_vec[0]
    phi = angle_vec[1:]

    # theta_V = 0
    _, rho = phase_sequence(0.0, phi, psi_W, K)
    prob_theta0 = probability_target(rho)

    # theta_V = theta_1
    _, rho = phase_sequence(theta_1, phi, psi_W, K)
    prob_theta1 = probability_target(rho)

    if FLAG_kl:
        error = 0.25*np.exp(-m*kl_divergence(prob_theta0, prob_theta1))
    else:
        if FLAG_optimize_gamma:
            gamma = optimize_chernoff_information(prob_theta0, prob_theta1)

            if FLAG_verbose:
                print('gamma=%f' % gamma)

        error = 0.5*np.exp(m * chernoff_information(gamma, prob_theta0, prob_theta1))

    return error
