#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=gpu-2080ti
#SBATCH --mem=100
#SBATCH --output=hostname_%j.out
#SBATCH --error=hostname_%j.err
#SBATCH --time=0-00:01

python run.py
