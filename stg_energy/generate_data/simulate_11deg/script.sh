#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --partition=cpu-short
#SBATCH --mem=10000
#SBATCH --output=/mnt/qb/work/macke/mdeistler57/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster_no_noise/01_simulate_11deg/outfiles/out_%j.out
#SBATCH --error=/mnt/qb/work/macke/mdeistler57/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster_no_noise/01_simulate_11deg/outfiles/out_%j.err
#SBATCH --time=0-10:00

python simulate_11deg.py
