# Imports
import os, sys, types
import pathlib
from importlib import reload
import pickle

import numpy as np
import matplotlib.pyplot as plt


def plot_hbcd_trends(delta_array_plot, theta1_array_plot, K_array_plot, depth_array_plot=None,
                    lower_bound=None, lower_bound_num=None, FLAG_monotonic=True,
                    savefile='numerical_bounds_K_hcd.png'):
    """
    Inputs:
        :param theta1_array_plot: list of theta1_arrays
        :param K_array_plot: list of K_arrays
        :param lower_bound: list with [array of theta1, array of lower bound values]
        :param lower_bound_num: list with [array of theta1, array of numerical lower bound values]

    :return:
    """
    color_plot = ['r', 'b', 'g', 'orange', 'mediumturquoise', 'hotpink', 'c', 'm', 'y']
    marker_plot = ['-x', '-o', '-s', '-d', '-^', '-p']
    n_delta = len(delta_array_plot)

    if n_delta != len(K_array_plot) or n_delta != len(theta1_array_plot):
        raise ValueError("Unequal lengths of delta, theta1 and K arrays")

    if lower_bound is None:
        theta1_array = theta1_array_plot[0]
        lb_values = 1 / np.sqrt(2 * (1 - np.cos(theta1_array)))
        lower_bound = [theta1_array, lb_values]

    if lower_bound_num is None:
        theta1_array = theta1_array_plot[0]
        lb_num_values = 2 / theta1_array
        lower_bound_num = [theta1_array, lb_num_values]

    plt.figure(figsize=(10,7))
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    for ind_delta in range(n_delta):
        delta = delta_array_plot[ind_delta]
        K_array_d = K_array_plot[ind_delta]
        theta1_array = theta1_array_plot[ind_delta]

        # Make sure K_array_d is monotonically increasing
        if FLAG_monotonic:
            for ind in reversed(range(1, len(theta1_array))):
                if K_array_d[ind] < K_array_d[ind-1]:
                    K_array_d[ind-1] = K_array_d[ind]

        if ind_delta < 3:
            plt.plot(theta1_array, K_array_d, marker_plot[ind_delta], c=color_plot[ind_delta], markersize=10,
                     markerfacecolor='none',
                     label=r'$\mathcal{P}_S$ ($\epsilon=%.3f$)' % delta)
        else:
            depth_query = depth_array_plot[ind_delta]
            plt.plot(theta1_array, K_array_d, marker_plot[ind_delta], c=color_plot[ind_delta], markersize=10,
                     markerfacecolor='none',
                     label=r'$\mathcal{P}_M$ ($d=%d, \, \epsilon=%.3f$)' % (depth_query, delta))

    plt.plot(lower_bound[0], lower_bound[1], '--', c='k', markersize=6, linewidth=1.5, label=r'$\mathcal{P}_S$ Lower Bound')
    plt.plot(lower_bound_num[0], lower_bound_num[1], ':', c='k', linewidth=1.5, label=r'$N \alpha$=constant')

    # Lower bound for SQL
    theta1_array = theta1_array_plot[0]
    lb_sql_values = 2 / theta1_array**2

    plt.plot(theta1_array, lb_sql_values, '-.', c='k', linewidth=1.5, label=r'$N \alpha^2$=constant')

    plt.xlabel(r'$\alpha$')
    plt.ylabel('N')
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks([0.025, 0.05, 0.1, 0.2, 0.4, 0.8], [r'$2.5 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
                                                   r"$2 \times 10^{-1}$", r"$4 \times 10^{-1}$", r"$8 \times 10^{-1}$"])
    #plt.yticks([2, 5, 10, 20, 40, 80], ['2', '5', '10', '20', '40', '80'])
    plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4], [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r"$10^{3}$", r"$10^{4}$"])

    # plt.legend(loc='upper right', bbox_to_anchor=(1.52, 1.02))

    #plt.legend(loc='upper right', ncol=2)
    plt_leg = plt.legend(loc='lower left', ncol=2, bbox_to_anchor=(0.05,-0.45))
    plt_leg.get_frame().set_edgecolor('gray')
    plt_leg.get_frame().set_linewidth(0.5)
    plt.savefig(savefile, bbox_inches='tight', pad_inches=0, dpi=300)


def plot_hbcd_trends_talk(delta_array_plot, theta1_array_plot, K_array_plot, depth_array_plot=None,
                         lower_bound=None, lower_bound_num=None, FLAG_monotonic=True,
                         savefile='numerical_bounds_K_hcd.png'):
    """
    Inputs:
        :param theta1_array_plot: list of theta1_arrays
        :param K_array_plot: list of K_arrays
        :param lower_bound: list with [array of theta1, array of lower bound values]
        :param lower_bound_num: list with [array of theta1, array of numerical lower bound values]

    :return:
    """
    color_plot = ['r', 'b', 'g', 'orange', 'mediumturquoise', 'hotpink', 'c', 'm', 'y']
    marker_plot = ['-x', '-o', '-s', '-d', '-^', '-p']
    n_delta = len(delta_array_plot)

    if n_delta != len(K_array_plot) or n_delta != len(theta1_array_plot):
        raise ValueError("Unequal lengths of delta, theta1 and K arrays")

    if lower_bound is None:
        theta1_array = theta1_array_plot[0]
        lb_values = 1 / np.sqrt(2 * (1 - np.cos(theta1_array)))
        lower_bound = [theta1_array, lb_values]

    if lower_bound_num is None:
        theta1_array = theta1_array_plot[0]
        lb_num_values = 2 / theta1_array
        lower_bound_num = [theta1_array, lb_num_values]

    # To force xlabels and ylabels to use the right defined Font
    # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
    from matplotlib import rc
    rc('text.latex', preamble=r'\usepackage{sfmath}')

    plt.figure(figsize=(12,9))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 26
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['text.usetex'] = True

    for ind_delta in range(n_delta):
        delta = delta_array_plot[ind_delta]
        K_array_d = K_array_plot[ind_delta]
        theta1_array = theta1_array_plot[ind_delta]

        # Make sure K_array_d is monotonically increasing
        if FLAG_monotonic:
            for ind in reversed(range(1, len(theta1_array))):
                if K_array_d[ind] < K_array_d[ind-1]:
                    K_array_d[ind-1] = K_array_d[ind]

        if ind_delta < 3:
            plt.plot(theta1_array, K_array_d, marker_plot[ind_delta], c=color_plot[ind_delta], markersize=10,
                     markerfacecolor='none',
                     label=r'$\Sigma_S$ ($\epsilon=%.3f$)' % delta)
        else:
            depth_query = depth_array_plot[ind_delta]
            plt.plot(theta1_array, K_array_d, marker_plot[ind_delta], c=color_plot[ind_delta], markersize=10,
                     markerfacecolor='none',
                     label=r'$\Sigma_M$ ($d=%d, \, \epsilon=%.3f$)' % (depth_query, delta))

    plt.plot(lower_bound[0], lower_bound[1], '--', c='k', markersize=6, linewidth=1.5, label=r'$\Sigma_S$ Lower Bound')
    plt.plot(lower_bound_num[0], lower_bound_num[1], ':', c='k', linewidth=1.5, label=r'$N \alpha$=constant')

    # Lower bound for SQL
    theta1_array = theta1_array_plot[0]
    lb_sql_values = 2 / theta1_array**2

    plt.plot(theta1_array, lb_sql_values, '-.', c='k', linewidth=1.5, label=r'$N \alpha^2$=constant')

    # Text annotations
    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 22,
            }
    plt.text(0.04, 2e3, 'Multi-shot protocol (SQL)', fontdict=font)
    plt.text(0.05, 7, 'Sequential protocol \n (Heisenberg scaling)', ha='center', va='center', fontdict=font)

    plt.xlabel(r'$\alpha$')
    plt.ylabel('N')
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks([0.025, 0.05, 0.1, 0.2, 0.4, 0.8], [r'$2.5 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
                                                   r"$2 \times 10^{-1}$", r"$4 \times 10^{-1}$", r"$8 \times 10^{-1}$"])
    #plt.yticks([2, 5, 10, 20, 40, 80], ['2', '5', '10', '20', '40', '80'])
    plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4], [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r"$10^{3}$", r"$10^{4}$"])

    plt_leg = plt.legend(loc='upper right')
    #plt_leg = plt.legend(loc='upper right', bbox_to_anchor=(1.52, 1.02))
    #plt_leg = plt.legend(loc='upper right', ncol=2)
    #plt_leg = plt.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.05,-0.45))
    plt_leg.get_frame().set_edgecolor('gray')
    plt_leg.get_frame().set_linewidth(0.5)
    plt.savefig(savefile, bbox_inches='tight', pad_inches=0.05, dpi=600)


def plot_hbcd_trends_paper(delta_array_plot, theta1_array_plot, K_array_plot, depth_array_plot=None,
                         lower_bound=None, lower_bound_num=None, FLAG_monotonic=True,
                         savefile='numerical_bounds_K_hcd.png'):
    """
    Inputs:
        :param theta1_array_plot: list of theta1_arrays
        :param K_array_plot: list of K_arrays
        :param lower_bound: list with [array of theta1, array of lower bound values]
        :param lower_bound_num: list with [array of theta1, array of numerical lower bound values]

    :return:
    """
    color_plot = ['r', 'b', 'g', 'orange', 'mediumturquoise', 'hotpink', 'c', 'm', 'y']
    marker_plot = ['-x', '-o', '-s', '-d', '-^', '-p']
    n_delta = len(delta_array_plot)

    if n_delta != len(K_array_plot) or n_delta != len(theta1_array_plot):
        raise ValueError("Unequal lengths of delta, theta1 and K arrays")

    if lower_bound is None:
        theta1_array = theta1_array_plot[0]
        lb_values = 1 / np.sqrt(2 * (1 - np.cos(theta1_array)))
        lower_bound = [theta1_array, lb_values]

    if lower_bound_num is None:
        theta1_array = theta1_array_plot[0]
        lb_num_values = 2 / theta1_array
        lower_bound_num = [theta1_array, lb_num_values]

    # To force xlabels and ylabels to use the right defined Font
    # Ref: https://stackoverflow.com/questions/11611374/sans-serif-font-for-axes-tick-labels-with-latex
    from matplotlib import rc
    rc('text.latex', preamble=r'\usepackage{sfmath}')

    fig = plt.figure(figsize=(12,6.5))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['text.usetex'] = True

    for ind_delta in range(3):
        delta = delta_array_plot[ind_delta]
        K_array_d = K_array_plot[ind_delta]
        theta1_array = theta1_array_plot[ind_delta]

        # Make sure K_array_d is monotonically increasing
        if FLAG_monotonic:
            for ind in reversed(range(1, len(theta1_array))):
                if K_array_d[ind] < K_array_d[ind-1]:
                    K_array_d[ind-1] = K_array_d[ind]

        plt.plot(theta1_array, K_array_d, marker_plot[ind_delta], c=color_plot[ind_delta], markersize=10,
                 markerfacecolor='none',
                 label=r'$\Sigma_S$ ($\epsilon=%.3f$)' % delta)

    # Sequential lower bound
    plt.plot(lower_bound[0], lower_bound[1], '--', c='k', markersize=6, linewidth=1.5, label=r'$\Sigma_S$ Lower Bound')

    for ind_delta in range(3,n_delta):
        delta = delta_array_plot[ind_delta]
        K_array_d = K_array_plot[ind_delta]
        theta1_array = theta1_array_plot[ind_delta]

        # Make sure K_array_d is monotonically increasing
        if FLAG_monotonic:
            for ind in reversed(range(1, len(theta1_array))):
                if K_array_d[ind] < K_array_d[ind-1]:
                    K_array_d[ind-1] = K_array_d[ind]

        depth_query = depth_array_plot[ind_delta]
        plt.plot(theta1_array, K_array_d, marker_plot[ind_delta], c=color_plot[ind_delta], markersize=10,
                 markerfacecolor='none',
                 label=r'$\Sigma_M$ ($\epsilon=%.3f$)' % delta)

    plt.plot(lower_bound_num[0], lower_bound_num[1], ':', c='k', linewidth=1.5, label=r'$N \alpha$=constant')

    # Lower bound for SQL
    theta1_array = theta1_array_plot[0]
    lb_sql_values = 2 / theta1_array**2

    plt.plot(theta1_array, lb_sql_values, '-.', c='k', linewidth=1.5, label=r'$N \alpha^2$=constant')

    # Text annotations
    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 22,
            }
    plt.text(0.26, 2e2, 'Multi-shot \n protocol (SQL)',  ha='center', va='center', fontdict=font)
    plt.text(0.045, 7, 'Sequential protocol \n (Heisenberg scaling)', ha='center', va='center', fontdict=font)

    plt.xlabel(r'$\alpha$')
    plt.ylabel('N')
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks([0.025, 0.05, 0.1, 0.2, 0.4, 0.8], [r'$2.5 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
                                                   r"$2 \times 10^{-1}$", r"$4 \times 10^{-1}$", r"$8 \times 10^{-1}$"])
    #plt.yticks([2, 5, 10, 20, 40, 80], ['2', '5', '10', '20', '40', '80'])
    plt.yticks([1e0, 1e1, 1e2, 1e3, 1e4], [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r"$10^{3}$", r"$10^{4}$"])

    #plt_leg = plt.legend(loc='upper right', ncol=2)
    #plt_leg = plt.legend(loc='upper right', bbox_to_anchor=(1.52, 1.02))
    #plt_leg = plt.legend(loc='upper right', ncol=2)
    plt_leg = plt.legend(loc='upper right', ncol=2, bbox_to_anchor=(1.01,1.02), labelspacing=0.25, fontsize=19)
    plt_leg.get_frame().set_edgecolor('gray')
    plt_leg.get_frame().set_linewidth(0.5)

    fig.gca().xaxis.set_label_coords(0.5, -0.05)

    plt.savefig(savefile, bbox_inches='tight', pad_inches=0.05, dpi=600)