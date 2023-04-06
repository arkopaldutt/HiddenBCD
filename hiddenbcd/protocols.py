# Import different paths
import numpy as np
import pickle

# local imports
from .estimators import likelihood_ratio_test, error_majority_vote, error_likelihood_ratio_test
from .queries import phase_sequence, probability_target, repeated_search_initial_conditions2, \
    repeated_search_numerical_length


# Relevant function for multi-shot protocol
def get_nshots_for_error_success(theta1, K, phi_sol, epsilon, FLAG_odd_m=False,
                                 type_estimator='lrt', FLAG_kl=True, gamma=0.5, FLAG_optimize_gamma=False):
    """
    Searching for only odd values of m allows us to avoid "equal decision" for m/2 even

    :param theta1:
    :param K:
    :param phi_sol:
    :param epsilon:
    :return:
    """
    # We allow only two types of estimators at the moment: majority vote or likelihood ratio test
    if type_estimator not in ['majority', 'lrt']:
        raise RuntimeError("Check type of estimator being used!")

    if type_estimator == 'majority':
        m = 1
        while error_majority_vote(m, phi_sol, theta1, K) > epsilon:
            if FLAG_odd_m:
                m += 2
            else:
                m += 1

    elif type_estimator == 'lrt':
        m = 1
        while error_likelihood_ratio_test(m, phi_sol, theta1, K, FLAG_kl=FLAG_kl,
                                          gamma=gamma, FLAG_optimize_gamma=FLAG_optimize_gamma) > epsilon:
            if FLAG_odd_m:
                m += 2
            else:
                m += 1

    return m


def experiment_multishot_protocol(m_shots, theta1, prob_theta0, prob_theta1, N_sets, FLAG_verbose=False):
    FLAG_success = True

    # Choose the true theta value for each experiment
    theta_truth_array = np.random.choice([0.0, theta1], size=N_sets, p=[0.5, 0.5])

    n_trials = 0
    while FLAG_success is True and n_trials < N_sets:
        # sample randomly from theta0 or alpha
        theta_truth = theta_truth_array[n_trials]

        # run experiment with 'm' shots
        if theta_truth > theta1/2:
            prob_binomial = prob_theta1
        else:
            prob_binomial = prob_theta0

        # sample 'm' shots from binomial distribution
        outcomes = np.random.binomial(m_shots, prob_binomial[1])

        # get the estimate
        theta_est = likelihood_ratio_test(m_shots, outcomes, theta1, prob_theta0, prob_theta1)

        # check if estimate matches the truth
        if not np.isclose(theta_est, theta_truth):
            FLAG_success = False
        n_trials += 1

        if FLAG_verbose:
            print('Success: %d/%d' %(n_trials, N_sets))

    return FLAG_success


# Describes procedure for multishot protocol to obtain number of shots required for a given depth K to get eps success
def get_nshots_for_error_success_experiment(theta1, K, phi_sol, epsilon, N_sets=45, FLAG_verbose=False):
    """
    Returns number of shots for which we succeed in discriminating "45 times" with the LRT

    TODO: Add an assertion that N_sets are satisfactory for achieving epsilon error
    """
    if epsilon < 0.05 and N_sets < 45:
        raise ValueError("Incompatible values of N_sets and epsilon")

    # get a starting value from the lower bound computation
    m = get_nshots_for_error_success(theta1, K, phi_sol, epsilon, type_estimator='lrt', FLAG_kl=True)

    if FLAG_verbose:
        print('Lower bound on shots = %d' % m)

    # determine the probabilities of target qubit given thetaV value
    if len(phi_sol) != K + 2:
        raise ValueError("Check length of angle vector")

    psi_W = phi_sol[0]
    phi = phi_sol[1:]

    # theta_V = 0
    _, rho = phase_sequence(0.0, phi, psi_W, K)
    prob_theta0 = probability_target(rho)

    # theta_V = theta_1
    _, rho = phase_sequence(theta1, phi, psi_W, K)
    prob_theta1 = probability_target(rho)

    # Set initial FLAG
    FLAG_success = False

    # If m=1, check if the error probability itself is as required
    if m == 1:
        p_error = error_majority_vote(1, phi_sol, theta1, K)
        if p_error <= epsilon:
            FLAG_success = True

    while FLAG_success is False:
        FLAG_success = experiment_multishot_protocol(m, theta1, prob_theta0, prob_theta1, N_sets=N_sets,
                                                     FLAG_verbose=FLAG_verbose)
        m += 1

    return m


# Describes job for multi-shot strategy
def search_multishot_protocol(theta1_array, depth, epsilon, n_reps, type_estimator='majority',
                              FLAG_kl=True, FLAG_optimize_gamma=False, FLAG_odd_m=False,
                              savefile='solutions_hcd_multishot_depth_epsilon.pickle', FLAG_verbose=True,
                              FLAG_experiment = False, N_sets=45,
                              FLAG_logger=True, log_run_filename='keeping_up.txt'):
    """
    We only allow two types of error computations at the moment (upper bound and lower bound) for the ML estimator

    We need to replace that with the "experimental" case as well

    Inputs:
    :param theta1_array:
    :param depth:
    :param epsilon:
    :param n_reps:
    :param type_estimator:
    :param FLAG_odd_m:
    :param savefile:
    :param FLAG_verbose:
    :param FLAG_logger:
    :param log_run_filename:
    :return:
    """
    # Solution vectors
    m_sol_d = np.zeros(len(theta1_array))
    N_sol_d = np.zeros(len(theta1_array))
    phi_sol_d = []

    print('Have chosen estimator: %s' % type_estimator)
    # Loop over theta1
    for ind in range(len(theta1_array)):
        theta1_temp = theta1_array[ind]

        if FLAG_verbose:
            print('Starting %d, alpha=%1.4f' % (ind, theta1_temp))

        # Get an optimal solution
        phi_opt, _ = repeated_search_initial_conditions2(theta1_temp, depth, n_reps)

        # Determine the number of shots required using this phi_opt to achieve epsilon error
        if FLAG_experiment:
            # Lets conduct an actual experiment and this is only done for
            print('Experiment study for the elusive m begins!')
            m_sol = get_nshots_for_error_success_experiment(theta1_temp, depth, phi_opt, epsilon, N_sets=N_sets,
                                                            FLAG_verbose=FLAG_verbose)
        else:
            # Skip experiment and just determine the
            m_sol = get_nshots_for_error_success(theta1_temp, depth, phi_opt, epsilon, type_estimator=type_estimator,
                                                 FLAG_odd_m=FLAG_odd_m, FLAG_kl=FLAG_kl,
                                                 FLAG_optimize_gamma=FLAG_optimize_gamma)
        N_sol = m_sol*depth

        m_sol_d[ind] = m_sol
        N_sol_d[ind] = N_sol
        phi_sol_d.append(phi_opt)

        if FLAG_verbose:
            print('Done with %d, alpha=%1.4f, m=%d' % (ind, theta1_temp, m_sol))

        if FLAG_logger:
            f_log = open(log_run_filename, "a+")
            # Iteration, Estimate, Error
            f_log.write("%f, %d, %d, %d" % (theta1_temp, depth, m_sol, N_sol))
            for ind_phi in range(len(phi_opt)):
                f_log.write(", %f" % phi_opt[ind_phi])

            f_log.write("\n")
            f_log.close()

    with open(savefile, 'wb') as handle:
        pickle.dump([m_sol_d, N_sol_d, phi_sol_d], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return m_sol_d, N_sol_d, phi_sol_d


# Describes job for sequential protocol
def search_seq_protocol_delta(theta1_array, delta, n_reps, savefile='solutions_hcd_delta.pickle', FLAG_logger=True,
                              log_run_filename='keeping_up.txt'):
    # Solution vectors
    K_sol_d = np.zeros(len(theta1_array))
    phi_sol_d = []

    # Initial length
    K0 = 1

    # Loop over theta1
    for ind in range(len(theta1_array)):
        theta1_temp = theta1_array[ind]
        K_sol, phi_sol = repeated_search_numerical_length(theta1_temp, delta, n_reps, K0=K0)

        K_sol_d[ind] = K_sol
        phi_sol_d.append(phi_sol)

        # Update initial length
        K0 = int(np.amax([np.floor(K0 / 2), 1]))

        print('Done with %d' % ind)

        if FLAG_logger:
            f_log = open(log_run_filename, "a+")
            # Iteration, Estimate, Error
            f_log.write("%f, %d" % (theta1_temp, K_sol))
            for ind_phi in range(len(phi_sol)):
                f_log.write(", %f" % phi_sol[ind_phi])

            f_log.write("\n")
            f_log.close()

    with open(savefile, 'wb') as handle:
        pickle.dump([K_sol_d, phi_sol_d], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return K_sol_d, phi_sol_d
