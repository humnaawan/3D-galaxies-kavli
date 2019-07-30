#!/bin/bash

features_file='/Users/humnaawan/repos/3D-galaxies-kavli/data/tng_lowres_2d_summary/features_20.csv'
shape_datapath='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/tng-100/'
base_outdir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/'
# ------------------------------------------------------------------------------
# classification
# no masses
outdir=${base_outdir}'rf_output/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir}
# ------------------------------------------------------------------------------
# regression
# no masses
outdir=${base_outdir}'rf_output/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --regress
