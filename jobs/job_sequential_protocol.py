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

from hiddenbcd.protocols import search_seq_protocol_delta


# Get arguments for the run
parser = argparse.ArgumentParser(description='Numerical Run')
parser.add_argument('--delta', type=float, default=0.05, metavar='N')
parser.add_argument('--expt_id', type=int, default=1, metavar='N')
args = parser.parse_args()

# Range of theta1
theta1_array = np.linspace(0.025, np.pi / 4, 64)
theta1_array = np.flip(theta1_array)    # Go in ascending order of difficulty

# Number of repetitions
n_reps = 5

# Start of job
delta = args.delta  # delta = 0.05 (default)
save_filename = 'solutions_hcd_delta_%f.pickle' % delta
log_run_filename = 'solutions_hcd_delta_%f.txt' % delta

print('Working on delta=%f' % delta)
K_array_d5, phi_sol_d5 = search_seq_protocol_delta(theta1_array, delta, n_reps, savefile=save_filename,
                                                   log_run_filename=log_run_filename)
