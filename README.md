# Hidden Binary Channel Discrimination (HBCD)

(Code accompanying the paper [Power of sequential protocols in hidden quantum channel discrimination](https://arxiv.org/abs/2304.02053))
## Introduction

In this work, we study the problem of learning an unknown quantum channel $C$ acting on a system that we cannot directly measure, arbitrarily control nor initialize reliably. This setting appears naturally in quantum logic detection and quantum non-demolition measurements, and engineered systems.

Concretly, we consider an unknown channel given by $C = \exp(i \theta_C \sigma_x)$ with an unknown phase $\theta_C$ taking values of $0$ or $\alpha \in (0,2\pi)$. The goal is to then discriminate the value of $\theta_C$ with the fewest measuremnets. Towards achieving this goal, we specify sequential and multi-shot protocols utilizing queries as shown in the Figure below. 

<p align="center">
  <img src="https://github.com/arkopaldutt/HiddenBCD/blob/main/figures/hbcd_query.pdf" width=60% height=60%>
</p>

See the paper for more details on how the queries are specified and optimized. We show that sequential protocols outperform multi-shot protocols by achieving a lower error probability on HBCD problems with fewer queries and at the Heisenberg limit.

<p align="center">
  <img src="https://github.com/arkopaldutt/HiddenBCD/blob/main/figures/comparison_hbcd_protocols.pdf" width=60% height=60%>
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

To run this package, we recommend installing the requirements specified in [hbcd_requirements.txt](https://github.com/)

## Citing this repository

To cite this repository please include a reference to our [paper](https://arxiv.org/abs/2304.02053).
