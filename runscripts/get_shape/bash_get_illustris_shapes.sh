#!/bin/bash

base_dir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/'
test=0  # 0 = no
z0=0  # 0 = no
if [ $test == 1 ];
then
    data_dir=${base_dir}'test_illustris/'
    outdir=${base_dir}'test_illustris_shape100/'
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_shape/run_get_shapes.py \
                    --data_dir=${data_dir} --illustris --z=0.0 --test \
                    --outdir=${outdir} --post_process_only
fi

if [ $test == 0 ];
then
    if [ ${z0} == 1 ];
    then
      data_dir=${base_dir}'illustris_z0.0/'
      outdir=${base_dir}'illustris_z0.0_shape100/'
      python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_shape/run_get_shapes.py \
                      --data_dir=${data_dir} --illustris --z=0.0 \
                      --outdir=${outdir} --post_process_only --extended_rstar
      #
      outdir=${base_dir}'illustris_z0.0_shape50/'
      python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_shape/run_get_shapes.py \
                      --data_dir=${data_dir} --illustris --z=0.0 --rdecider=50 \
                      --outdir=${outdir} --post_process_only --extended_rstar

    fi
    if [ ${z0} == 0 ];
    then
        data_dir=${base_dir}'illustris_z0.4/'
        outdir=${base_dir}'illustris_z0.4_shape100/'
        python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_shape/run_get_shapes.py \
                        --data_dir=${data_dir} --illustris --z=0.4 \
                        --outdir=${outdir} --post_process_only
        #
        outdir=${base_dir}'illustris_z0.4_shape50/'
        python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_shape/run_get_shapes.py \
                        --data_dir=${data_dir} --illustris --z=0.4 --rdecider=50 \
                        --outdir=${outdir} --post_process_only
    fi
fi
