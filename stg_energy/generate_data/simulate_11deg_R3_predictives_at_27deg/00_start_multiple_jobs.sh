#!/usr/bin/env bash

n_jobs=12

for ((d=0; d<$n_jobs; d++))
do
    sbatch script.sh
    sleep 2
done
