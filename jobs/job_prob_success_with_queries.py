# Imports
import os, sys, types
import pathlib
import argparse

import numpy as np
import pickle

# Local package imports
# Update this with setup & develop later
PROJECT_PATH = str(pathlib.Path().resolve().parent)
sys.path.append(PROJECT_PATH)

import qspsensing.hidden_channel_discrimination as qsp_hcd

N_array = np.arange(2, 30)
alpha = 0.25
n_reps = 10

p0_theta0 = np.zeros(len(N_array))
p0_theta1 = np.zeros(len(N_array))
p_error = np.zeros(len(N_array))
p_success = np.zeros(len(N_array))
phi_opt = []

for ind in range(len(N_array)):
    K = N_array[ind]

    # Get an optimal solution
    phi_sol, _ = qsp_hcd.repeated_search_initial_conditions2(alpha, K, n_reps)
    phi_opt.append(phi_sol)

    # Compute the relevant probabilities
    psi_W = phi_sol[0]
    phi = phi_sol[1:]

    # theta_V = 0
    _, rho = qsp_hcd.shadow_qsp_sequence(0.0, phi, psi_W, K)
    prob_theta0 = qsp_hcd.probability_target(rho)
    p0_theta0[ind] = prob_theta0[0]

    # theta_V = theta_1
    _, rho = qsp_hcd.shadow_qsp_sequence(alpha, phi, psi_W, K)
    prob_theta1 = qsp_hcd.probability_target(rho)
    p0_theta1[ind] = prob_theta1[0]

    # Error
    p_error[ind] = qsp_hcd.error_majority_vote(1, phi_sol, alpha, K)

    # Success
    p_success[ind] = 1 - p_error[ind]

    print('Done with K=%d' % K)

# Dump to a pickle file
ds = {'phi_opt': phi_opt, 'N': N_array, 'p0_theta0': p0_theta0, 'p0_theta1': p0_theta1,
      'p_error': p_error, 'p_success': p_success, 'alpha': alpha}

savefile = 'prob_trend_with_N_hbcd_alpha_0_25.pickle'

with open(savefile, 'wb') as handle:
    pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)