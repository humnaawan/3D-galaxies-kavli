#!/bin/bash

features_file='/Users/humnaawan/repos/3D-galaxies-kavli/data/tng_highres/xy/features_18.csv'
shape_datapath='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/tng-100/'
base_outdir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/'
# ------------------------------------------------------------------------------
# classification
# no masses
outdir=${base_outdir}'rf_output/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --high_res
# ------------------------------------------------------------------------------
# regression
# no masses
outdir=${base_outdir}'rf_output/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --regress --high_res
