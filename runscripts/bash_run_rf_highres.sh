#!/bin/bash
shape_datapath='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/tng-100/'
base_outdir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/'
# ------------------------------------------------------------------------------
# run  analysis for one projection;
features_file='/Users/humnaawan/repos/3D-galaxies-kavli/data/tng_highres/xy/features_28.csv'
# no masses included; no a1a4; 3 classes
outdir=${base_outdir}'rf_tng_highres_we_3classes/'
# classification
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --high_res --plot_feature_dists \
                --no_2nd_order_feats --good_radius_feats
# regression
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --high_res \
                --no_2nd_order_feats --good_radius_feats --regress
# ------------------------------------------------------------------------------
# 2 classes; no masses included; no a1a4
outdir=${base_outdir}'rf_tng_highres_we_2classes/'
# classification
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --high_res \
                --no_2nd_order_feats --good_radius_feats \
                --prolate_vs_not
# ------------------------------------------------------------------------------
# now include a1a4
# no masses included; 3 classes
outdir=${base_outdir}'rf_tng_highres_we_3classes_wa1a4/'
# classification
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --high_res --plot_feature_dists \
                --good_radius_feats
# regression
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --high_res \
                --good_radius_feats --regress
# ------------------------------------------------------------------------------
# 2 classes; no masses included
outdir=${base_outdir}'rf_tng_highres_we_2classes_wa1a4/'
# classification
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                --features_file=${features_file} \
                --shape_datapath=${shape_datapath} \
                --outdir=${outdir} --high_res \
                --good_radius_feats \
                --prolate_vs_not
