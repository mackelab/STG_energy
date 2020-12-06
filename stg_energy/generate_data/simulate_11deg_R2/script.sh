#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --partition=cpu-long
#SBATCH --mem=20000
#SBATCH --output=/home/macke/mdeistler57/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/01_simulate_11deg_R2/outfiles/out_%j.out
#SBATCH --error=/home/macke/mdeistler57/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/01_simulate_11deg_R2/outfiles/out_%j.err
#SBATCH --time=0-10:00

python simulate_11deg.py
