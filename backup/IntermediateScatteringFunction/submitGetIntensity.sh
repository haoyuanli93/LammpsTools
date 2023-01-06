#!/bin/bash
#
#SBATCH --job-name=MDtest1 # Job name for allocation
#SBATCH --output=logFiles/%j.log # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=logFiles/%j.error # File to which STDERR will be written, %j inserts jobid
#SBATCH --partition=psanaq # Partition/Queue to submit job
#SBATCH ----gres=gpu:1  # Get 1 gpu for the job
#SBATCH --ntasks=4 # Total number of tasks

python ./getIntensityPerQ.py