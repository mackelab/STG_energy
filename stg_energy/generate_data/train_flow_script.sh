#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --partition=cpu-long
#SBATCH --mem=60000
#SBATCH --output=/home/macke/mdeistler57/Documents/STG_energy/stg_energy/generate_data/outfiles/out_%j.out
#SBATCH --error=/home/macke/mdeistler57/Documents/STG_energy/stg_energy/generate_data/outfiles/out_%j.err
#SBATCH --time=0-20:00

python train_flow_R3.py
