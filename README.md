# Hidden Binary Channel Discrimination (HBCD)

## Introduction

### Problem statement

### Query

## Protocols

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

To run this package, we recommend installing the requirements specified in [tf_requirements.txt](https://github.com/ichuang/pyqsp/blob/master/tf_requirements.txt)

## Citing this repository

To cite this repository please include a reference to our paper (arXiv link coming soon!).
