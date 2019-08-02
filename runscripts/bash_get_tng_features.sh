#!/bin/bash

summary_datapath='/Users/humnaawan/repos/3D-galaxies-kavli/data/tng_highres_z0.4_sum/xy/'
shape_datapath='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/tng-100/'

python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_features.py \
                --summary_datapath=${summary_datapath} \
                --shape_datapath=${shape_datapath} --outdir=${summary_datapath}
