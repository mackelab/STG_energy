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
