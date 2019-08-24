#!/bin/bash

base_dir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/'
data_dir=${base_dir}'tng-100_z0.4/'

#outdir=${base_dir}'tng-100_z0.4_shape100/'
#python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_shape/run_get_shapes.py \
#                --data_dir=${data_dir} --z=0.4 \
#                --outdir=${outdir} --post_process_only

outdir=${base_dir}'tng-100_z0.4_shape50/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_shape/run_get_shapes.py \
                --data_dir=${data_dir} --z=0.4 \
                --outdir=${outdir} --post_process_only --rdecider=50
