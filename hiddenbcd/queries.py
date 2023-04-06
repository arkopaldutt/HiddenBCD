# Import different paths
import numpy as np
import copy
import math
import random
import scipy
import scipy.linalg

import scipy.optimize
from functools import partial


# Pauli operators
si = np.array([ [1, 0], [0, 1] ])
sx = np.array([ [0, 1], [1, 0] ])
sy = np.array([ [0, -1j], [1j, 0] ])
sz = np.array([ [1, 0], [0, -1] ])
had = (1/np.sqrt(2))*np.array([ [1, 1], [1, -1] ])

# Kron
def kron(a, b):
    return np.matrix(scipy.linalg.kron(a, b))


# Helper functions
def Rx(phi):
    return np.cos(phi)*si + 1j*np.sin(phi)*sx


def Ry(phi):
    return np.cos(phi)*si + 1j*np.sin(phi)*sy


def Rz(phi):
    return np.cos(phi)*si + 1j*np.sin(phi)*sz


def print_clean(x):
    with np.printoptions(suppress=True):
        print(x)


# Define result of queries in terms of phase sequence
def phase_sequence(theta_V, phi, psi_W, K, rho_0=1 / 2 * np.eye(2), FLAG_verbose=False):
    """
    Computes the unitary operator from applying a Shadow QSP Sequence

    Assuming two-qubit circuit and initial state is 1/2(|0><0| + |1><1|) \otimes |0><0|
    Inputs:
        theta_V : scalar = unknown phase angle defining V = exp(i theta_V X)
        phi: array = angles of rotation on Rx
        psi_W: scalar = angle of rotation on Rz
        K: int = length (number of repetitions of block)

    Outputs:
        U_sqsp = resulting unitary from queries associated with phase sequence
        rho = evolved density matrix
    """

    if len(phi) != K + 1:
        raise ValueError("Check length of phi")

    # Initial state
    rho = kron(rho_0, np.array([[1, 0], [0, 0]]))

    # Apply phi_0
    U_sqsp = np.kron(si, Rx(phi[0]))
    rho = U_sqsp @ rho @ U_sqsp.conj().transpose()

    for k in range(1, K + 1):
        # Create block
        # - VI - CRz(psi_W) - IRx
        V = Rx(theta_V)
        VI = np.kron(V, si)

        CRz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(1j * psi_W), 0], [0, 0, 0, np.exp(-1j * psi_W)]])

        IRx = np.kron(si, Rx(phi[k]))

        U_block = IRx @ CRz @ VI

        if FLAG_verbose:
            print(k)
            print(U_block)

        # Update density matrix
        rho = U_block @ rho @ U_block.conj().transpose()

        # Update unitary
        U_sqsp = U_block @ U_sqsp

    #     # Apply IH
    #     IH = kron(si, had)

    #     rho = IH.conj().transpose() @ rho @ IH

    return U_sqsp, rho


def probability_target(rho):
    """
    Computes the probability of measuring the target qubit assuming rho is a two-qubit state

    Output:
        pvec = [p0,p1]
    """
    p0 = np.abs(rho[0, 0] + np.abs(rho[2, 2]))

    return np.array([p0, 1.0 - p0])


# Numerical optimization
def cost_function(angle_vec, theta_1, K):
    """
    Computes the above cost function (p(0|\theta_V=\theta_1) - p(0|\theta_V=0) - 1)**2

    Inputs:
        angle_vec: vector of angles arranged as (psi_W, phi)
        theta_1: assuming that theta_V takes values in (0,theta_1)
        K: number of repetitions
    """
    if len(angle_vec) != K + 2:
        raise ValueError("Check length of angle vector")

    psi_W = angle_vec[0]
    phi = angle_vec[1:]

    # theta_V = 0
    _, rho = phase_sequence(0.0, phi, psi_W, K)
    prob_theta0 = probability_target(rho)
    p_theta0 = prob_theta0[0]

    # theta_V = theta_1
    _, rho = phase_sequence(theta_1, phi, psi_W, K)
    prob_theta1 = probability_target(rho)
    p_theta1 = prob_theta1[0]

    return (p_theta1 - p_theta0 - 1.0) ** 2


def quasi_newton_solve(angle_vec_init, theta_1, K):
    """
    Solve the optimization problem with the Quasi-Newton method of L-FBGS-B

    Inputs:
        angle_vec_init (Required): initial condition to the solve
    """
    cost_func = partial(cost_function, theta_1=theta_1, K=K)
    res = scipy.optimize.minimize(cost_func, angle_vec_init,
                                  method='L-BFGS-B', jac=None, bounds=None)

    return res.x


# find a value of k for which a solution is found that satisfies given conditions
def search_numerical_length(theta_1, delta, K0=1):
    K = K0
    cost_sqsp = 1

    while cost_sqsp > delta:
        K += 1
        phi_init = np.zeros(K + 1)
        phi_init[0] = np.random.randn()
        phi_init[K] = np.random.randn()

        psi_W_init = np.random.randn()

        angle_vec_init = np.insert(phi_init, 0, psi_W_init, axis=0)

        phi_sol = quasi_newton_solve(angle_vec_init, theta_1, K)

        cost_sqsp = cost_function(phi_sol, theta_1, K)

    return K, phi_sol


def repeated_search_numerical_length(theta_1, delta, n_reps, K0=1):
    K_reps = np.zeros(n_reps, dtype=int)
    phi_reps = []

    for ind in range(n_reps):
        K_temp, phi_temp = search_numerical_length(theta_1, delta, K0=K0)
        K_reps[ind] = K_temp
        phi_reps.append(phi_temp)

    ind_sol = np.argmin(K_reps)
    K_sol = np.amin(K_reps)
    phi_sol = phi_reps[ind_sol]

    return K_sol, phi_sol


def repeated_search_initial_conditions(theta_1, K, n_reps):
    # K corresponds to number of blocks
    len_phi = K + 2

    # Current best guess
    cost_opt = 10
    phi_opt = np.zeros(len_phi)

    for ind in range(n_reps):
        # Each time we get a random IC
        phi_init = np.zeros(len_phi)

        phi_init[0] = np.random.randn()     # psi_W
        phi_init[1] = np.random.randn()     # phi_0
        phi_init[-1] = np.random.randn()    # phi_K

        phi_sol = quasi_newton_solve(phi_init, theta_1, K)

        cost_sqsp = cost_function(phi_sol, theta_1, K)

        # Compare obtained solution against current best solution
        if cost_sqsp < cost_opt:
            cost_opt = copy.copy(cost_sqsp)
            phi_opt = copy.deepcopy(phi_sol)

    return phi_opt, cost_opt


def repeated_search_initial_conditions2(theta_1, K, n_reps):
    # K corresponds to number of blocks
    len_phi = K + 2

    # Current best guess
    cost_opt = 10
    phi_opt = np.zeros(len_phi)

    # Construct all the initial conditions
    psiW_ics = np.random.randn(n_reps)
    phi0_ics = np.random.randn(n_reps)
    phik_ics = np.random.randn(n_reps)

    # Consider one where all phases are np.pi/4
    phi0_ics[0] = np.pi/4
    phik_ics[0] = np.pi / 4

    for ind in range(n_reps):
        # Each time we get a random IC
        phi_init = np.zeros(len_phi)

        phi_init[0] = psiW_ics[ind]     # psi_W
        phi_init[1] = phi0_ics[ind]     # phi_0
        phi_init[-1] = phik_ics[ind]    # phi_K

        phi_sol = quasi_newton_solve(phi_init, theta_1, K)

        cost_sqsp = cost_function(phi_sol, theta_1, K)

        # Compare obtained solution against current best solution
        if cost_sqsp < cost_opt:
            cost_opt = copy.copy(cost_sqsp)
            phi_opt = copy.deepcopy(phi_sol)

    return phi_opt, cost_opt
