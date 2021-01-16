#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --partition=cpu-long
#SBATCH --mem=100000
#SBATCH --output=/home/macke/mdeistler57/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/simulate_11deg_R3_predictives_at_27deg_notau_078/outfiles/out_%j.out
#SBATCH --error=/home/macke/mdeistler57/Documents/STG_energy/results/simulation_data_Tube_MLslurm_cluster/simulate_11deg_R3_predictives_at_27deg_notau_078/outfiles/out_%j.err
#SBATCH --time=0-12:00

python simulate_11deg.py
