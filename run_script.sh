#!/bin/bash


#SBATCH --job-name="musicgan" #Name of the job which appears in squeue
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lfb6ek@virginia.edu
#
#SBATCH --error="musicgan.err"                    # Where to write std err
#SBATCH --output="musicgan.out"                # Where to write stdout
#SBATCH --nodelist=ai05
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cudnn; module load cuda-toolkit-9.0; module load anaconda3
source activate env
