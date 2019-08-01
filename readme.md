This repo hosts code for exploring the connection between 3D shapes and 2D projections of galaxies. The main code is in [`d3g2d`](https://github.com/humnaawan/3D-galaxies-kavli/tree/master/d3g2d) while the [`runscripts`](https://github.com/humnaawan/3D-galaxies-kavli/tree/master/runscripts) folder contains the scripts to use the code in `d3g2d` to download data, get 3D shapes, and run classification/regression analysis.

This work has initiated as a part of the [Kavli Summer Program In Astrophysics 2019](https://kspa.soe.ucsc.edu/2019 ). The code to get shapes is based on Hongyu Li's [code]( https://github.com/HongyuLi2016/illustris-tools ); some of its outputs are included in his [paper](https://arxiv.org/abs/1709.03345 ).

----
To set up the code in development mode, clone the repo, cd into the cloned folder, and run
```
pip install --user -e .
```
