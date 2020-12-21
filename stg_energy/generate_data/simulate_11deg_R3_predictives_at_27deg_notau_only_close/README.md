# Generating simulated data

This folder simulates at 27deg. It simulates the data that were close at 11 degree. It is used to identify whehter there is a robust samples will be more or less temperature efficient.

Two steps:
1) Generate simulation outputs
```shell script
./00_start_multiple_runs.sh
```
2) Merge all files, and sort them into `valid` and `bad` simulations
```shell script
./01_merge_simulated_data.sh
```
