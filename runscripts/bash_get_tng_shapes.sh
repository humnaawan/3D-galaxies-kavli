#!/bin/bash

data_dir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/tng-100/'

python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_get_shapes.py \
                --data_dir=${data_dir} --rdecider=50
