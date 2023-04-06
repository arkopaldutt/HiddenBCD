# Imports
import os, sys, types
import pathlib
import argparse
import shutil

import pickle

import numpy as np
import math
import scipy
import scipy.linalg

import scipy.optimize

from functools import partial

import matplotlib.pyplot as plt

# Local package imports
# Update this with setup & develop later
PROJECT_PATH = str(pathlib.Path().resolve().parent)
sys.path.append(PROJECT_PATH)

from hiddenbcd.protocols import search_multishot_protocol


# Get arguments for the run
parser = argparse.ArgumentParser(description='Numerical Run')
parser.add_argument('--depth', type=int, default=2, metavar='N')
parser.add_argument('--epsilon', type=float, default=0.05, metavar='N')
parser.add_argument('--expt_id', type=int, default=1, metavar='N')
args = parser.parse_args()

# Range of theta1
theta1_array = np.linspace(0.025, np.pi / 4, 64)
theta1_array = np.flip(theta1_array)    # Go in ascending order of difficulty

# Number of repetitions
n_reps = 10

# Start of job
depth = args.depth
epsilon = args.epsilon
experiment_number = args.expt_id

# Set of other parameters for run
# Determined earlier
# For 90\% confidence, use N_sets=45 (epsilon=0.05), N_sets=91 (epsilon=0.025), N_sets=460 (epsilon=0.005)
# For 95\% confidence, use N_sets=59 (epsilon=0.05), N_sets=119 (epsilon=0.025), N_sets=598 (epsilon=0.005)
N_sets = 598

type_estimator = 'lrt'

save_filename = 'solutions_hcd_multishot_est_%s_d_%d_epsilon_%f_expt_%d.pickle' % (type_estimator, depth, epsilon, experiment_number)
log_run_filename = 'logfile_hcd_multishot_est_%s_d_%d_epsilon_%f_expt_%d.txt' % (type_estimator, depth, epsilon, experiment_number)

print('Working on epsilon=%f' % epsilon)
m_sol_d, N_sol_d, phi_sol_d = search_multishot_protocol(theta1_array, depth, epsilon, n_reps,
                                type_estimator=type_estimator, savefile=save_filename, FLAG_verbose=True,
                                FLAG_experiment=True, N_sets=N_sets,
                                FLAG_logger=True, log_run_filename=log_run_filename)
