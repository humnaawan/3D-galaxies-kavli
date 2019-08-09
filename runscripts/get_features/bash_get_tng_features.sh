#!/bin/bash

base_dir='/Users/humnaawan/repos/3D-galaxies-kavli/'


older=0

if [ $older == 1 ];
then
    summary_datapath=${base_dir}'data/tng_highres/xy/'

    shape_datapath=${base_dir}'outputs/tng-100_z0.4_shape100/'
    outdir=${base_dir}'outputs/tng-100_z0.4_shape100_features/'
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_features/get_features.py \
                    --summary_datapath=${summary_datapath} \
                    --shape_datapath=${shape_datapath} --outdir=${outdir} \
                    --data_tag='xy'

    shape_datapath=${base_dir}'outputs/tng-100_z0.4_shape50/'
    outdir=${base_dir}'outputs/tng-100_z0.4_shape50_features/'
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_features/get_features.py \
                    --summary_datapath=${summary_datapath} \
                    --shape_datapath=${shape_datapath} --outdir=${outdir} \
                    --rdecider=50 --data_tag='xy'
fi

if [ $older == 0 ];
then
    for proj in xy yz xz
    do
        summary_datapath=${base_dir}'data/sum_tng/'${proj}'/'
        for rdecider in 50 100
        do
            echo 'Running for '${proj}' for Rdecider = '${rdecider}
            shape_datapath=${base_dir}'outputs/tng-100_z0.4_shape'${rdecider}'/'
            outdir=${base_dir}'outputs/tng-100_z0.4_3proj_shape'${rdecider}'_features/'

            python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_features/get_features.py \
                            --summary_datapath=${summary_datapath} \
                            --shape_datapath=${shape_datapath} --outdir=${outdir} \
                            --data_tag=${proj} --summed_data --rdecider=${rdecider}
        done
    done
fi
