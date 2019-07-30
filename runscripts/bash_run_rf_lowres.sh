#!/bin/bash

features_file='/Users/humnaawan/repos/3D-galaxies-kavli/data/tng_lowres_2d_summary/features_19.csv'
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
# wm100
outdir=${base_outdir}'rf_output_wm100/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --m100
# wm200
outdir=${base_outdir}'rf_output_wm200/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --m200
# wm100 and m200
outdir=${base_outdir}'rf_output_mw100_wm200/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --m100 --m200
# ------------------------------------------------------------------------------
# regression
# no masses
outdir=${base_outdir}'rf_output/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --regress
# wm100
outdir=${base_outdir}'rf_output_wm100/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --m100 --regress
# wm200
outdir=${base_outdir}'rf_output_wm200/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --m200 --regress
# wm100 and m200
outdir=${base_outdir}'rf_output_mw100_wm200/'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --m100 --m200 --regress
