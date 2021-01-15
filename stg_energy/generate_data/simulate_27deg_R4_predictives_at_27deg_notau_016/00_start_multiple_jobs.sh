#!/usr/bin/env bash

n_jobs=6

for ((d=0; d<$n_jobs; d++))
do
    sbatch script.sh ${d}
    sleep 2
done
