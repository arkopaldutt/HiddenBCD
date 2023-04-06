#!/bin/bash

# Initialize the module command first
source /etc/profile

# Load Conda environment
conda activate hiddenbcd-env

#echo "My run number: " $1

# Call your script as you would from the command line passing $1 and $2 as arguments
#python job_sequential_protocol.py --delta=$1 --expt_id=1

python job_prob_success_with_queries.py

#python job_multishot_protocol.py --depth=5 --epsilon=0.005 --expt_id=31
#python job_multishot_protocol.py --depth=5 --epsilon=0.05 --expt_id=10
