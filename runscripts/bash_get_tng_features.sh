#!/bin/bash

base_dir='/Users/humnaawan/repos/3D-galaxies-kavli/'
summary_datapath=${base_dir}'data/tng_highres/xy/'

shape_datapath=${base_dir}'outputs/tng-100_z0.4_shape100/'
outdir=${base_dir}'outputs/tng-100_z0.4_shape100_features/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_features.py \
                --summary_datapath=${summary_datapath} \
                --shape_datapath=${shape_datapath} --outdir=${outdir} \
                --data_tag='xy'

shape_datapath=${base_dir}'outputs/tng-100_z0.4_shape50/'
outdir=${base_dir}'outputs/tng-100_z0.4_shape50_features/'
#python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_features.py \
#                --summary_datapath=${summary_datapath} \
#                --shape_datapath=${shape_datapath} --outdir=${outdir} \
#                --rdecider=50 --data_tag='xy'
