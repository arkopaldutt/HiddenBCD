# Hidden Binary Channel Discrimination (HBCD)

(Code accompanying the paper [Power of sequential protocols in hidden quantum channel discrimination](https://arxiv.org/abs/2304.02053))
## Introduction

In this work, we study the problem of learning an unknown quantum channel $C$ acting on a system that we cannot directly measure, cannot arbitrarily control nor initialize reliably. This setting appears naturally in quantum logic detection and quantum non-demolition measurements, and engineered systems.

Concretly, we tackle the problem of learning an unknown quantum channel $C$ acting on a hidden qubit that is a single-qubit rotation $C = \exp(i \theta_C \sigma_x)$ with the parameter $\theta_C$ being unknown and to be discriminated. The initial state $\rho_h$ of the hidden qubit is assumed to be the maximally mixed state. The true value of $\theta_C$ is either $0$ or a given $\alpha \in (0,2\pi)$ with equal probability. The hidden qubit is accessed by a measurement apparatus consisting of a single (target) qubit through a query that cannot directly manipulate the hidden system. A query involves $N$ serial applications of $C$ on the hidden qubit, a tunable controlled-rotation gate on to the target qubit, and single-qubit rotations on the target qubit. 

<p align="center">
  <img src="https://github.com/arkopaldutt/HiddenBCD/blob/main/figures/hbcd_query.png" width=60% height=60%>
</p>

See the paper for more details on how the queries are specified and optimized. The figure of merit of a protocol is the query complexity $N$ or the number of interactions with the hidden system required for accomplishing HBCD with error probability $\epsilon \in [0,1/2)$. We show that sequential protocols outperform multi-shot protocols by achieving a lower error probability on HBCD problems with fewer queries and at the Heisenberg limit.

<p align="center">
  <img src="https://github.com/arkopaldutt/HiddenBCD/blob/main/figures/comparison_hbcd_protocols.png" width=60% height=60%>
</p>

## Code design

The structure of the main code in `hiddenbcd` is as follows
* `queries.py` describes how queries are evaluated using corresponding phase sequences and optimized for a given HBCD problem.
* `estimators.py` defines the estimators of majority vote and Likelihood Ratio Test that can be used with different HBCD discrimination protocols. Functions for computing estimation errors are also provided.
* `protocols.py` describes how the sequential and mutl-shot protocols are used to solve HBCD problems.
* `utils_experiments.py` describes helper functions for carrying out experiments and post-processing experimental data.

Additionally, the `cases` directory includes the following files that demonstrate usage of the code
* `demo_hidden_channel_discrimination.py` works through how queries are defined, evaluated and optimized for a given HBCD problem.
* `demo_experiments.py` describes how sequential and multi-shot protocols are used in solving HBCD.

Finally, if you want to run larger jobs, there are scripts on running numerical experiments with the sequential and multi-shot protocol for solving HBCD in `jobs`. Post-processing of experimental data is illustrated in `process_jobs.py`. Data obtained in our numerical experiments and used in generating results for the paper can also be found here.
### Requirements

To run this package, we recommend installing the requirements specified in [hbcd_requirements.txt](https://github.com/arkopaldutt/HiddenBCD/blob/main/hbcd_requirements.txt)

## Citing this repository

To cite this repository please include a reference to our [paper](https://arxiv.org/abs/2304.02053).
