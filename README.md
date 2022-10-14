# Energy-efficient network activity from disparate circuit parameters

### Purpose of this repostitory

This repository contains all code and data to create the figures in the paper `Energy-efficient network activity from disparate circuit parameters` by Deistler, Macke*, Goncalves* (2022). If you only want to use the machine learning tools developed in this work, see the [sbi repository](https://github.com/mackelab/sbi) and the corresponding [tutorial](https://www.mackelab.org/sbi/tutorial/08_restriction_estimator/). If you only need the pyloric simulator, see [this repo](https://github.com/mackelab/pyloric).

### Installation
First, create a conda environment:
```
conda env create --file environment.yml
conda activate stg-energy
```
Then clone and install this package:
```
git clone git@github.com:mackelab/STG_energy.git
cd STG_energy
pip install -e .
```

Please note that all neural networks were trained on sbi v0.14.0. In v0.15.0, the training routine of `sbi` changed (z-score only using train data). Thus, training on a newer version give slightly different results.

### Structure of this repository
Roughly, the workflow for this work can be divided into three sections: (1) Running the pyloric simulator for many parameter sets, (2) Training the neural density estimator to approximate the posterior and (3) Generating plots.

(1) is implemented in `stg_energy/generate_data/simulate...` and was run on a compute cluster with SLURM.
(2) is implemented in `stg_energy/generate_data/train...`
(3) is implemented in `paper/`

### Commands to generate data and train the network
```
cd stg_energy/generate_data/simulate_11deg
python simulate_11deg.py
python 01_merge_simulated_data.py

python train_classifier.py

cd stg_energy/generate_data/simulate_11deg_R2
python simulate_11deg.py
python 01_merge_simulated_data.py

python train_classifier_R2.py

cd stg_energy/generate_data/simulate_11deg_R3
python simulate_11deg.py
python 01_merge_simulated_data.py

python train_flow_R3.py
```
### Git LFS

To store the data files, we use Git LFS.

### Citation
```
@article{deistler2022energy,
  title={Energy efficient network activity from disparate circuit parameters},
  author={Deistler, Michael and Macke, Jakob H and Gon{\c{c}}alves, Pedro J},
  journal={bioRxiv},
  pages={2021--07},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

### Contact
If you have any questions regarding the code, please contact `michael.deistler@uni-tuebingen.de`
