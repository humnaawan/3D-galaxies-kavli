#!/bin/bash

base_dir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/'
# ------------------------------------------------------------------------------
regress_wT_noa1a4=0
classify_wT_noa1a4=0
regress_wabc_noa1a4=0
classify_wabc_noa1a4=0
classify_wfe_noa1a4=1
# ------------------------------------------------------------------------------
# run  analysis for one projection;
rdecider=50
proj='xy'

features_file=${base_dir}'tng-100_z0.4_shape'${rdecider}'_features/features_41_high_res_'${proj}'_shape'${rdecider}'.csv'
base_outdir=${base_dir}'tng-100_z0.4_shape'${rdecider}'_rf_'

# no masses included; no a1a4; 3 classes
# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; no a1a4
shape_file=${base_dir}'tng-100_z0.4_shape'${rdecider}'/shape'${rdecider}'_data_295haloIds.csv'
if [ $regress_wT_noa1a4 == 1 ];
then
    outdir=${base_outdir}'regress_wT_noa1a4/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_file=${shape_file} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats --regress \
                    --rdecider=${rdecider} --just_triaxiality
fi
# ------------------------------------------------------------------------------
shape_file=${base_dir}'tng-100_z0.4_shape'${rdecider}'/shape'${rdecider}'_classes_T-based_0.7thres_295haloIds.csv'
# classify based on triaxiality; no masses; no a1a4
if [ $classify_wT_noa1a4 == 1 ];
then
    outdir=${base_outdir}'classify_wT_noa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file=${features_file} \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --no_2nd_order_feats --good_radius_feats \
                  --rdecider=${rdecider}
fi

#
# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; no a1a4
shape_file=${base_dir}'tng-100_z0.4_shape'${rdecider}'/shape'${rdecider}'_data_295haloIds.csv'
if [ $regress_wabc_noa1a4 == 1 ];
then
    outdir=${base_outdir}'regress_wabc_noa1a4/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_file=${shape_file} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats --regress \
                    --rdecider=${rdecider}
fi
# ------------------------------------------------------------------------------
shape_file=${base_dir}'tng-100_z0.4_shape'${rdecider}'/shape'${rdecider}'_classes_axis-ratios-based_295haloIds.csv'
# classify based on triaxiality; no masses; no a1a4
if [ $classify_wabc_noa1a4 == 1 ];
then
    outdir=${base_outdir}'classify_wabc_noa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file=${features_file} \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --no_2nd_order_feats --good_radius_feats \
                  --rdecider=${rdecider}
fi

# ------------------------------------------------------------------------------
shape_file=${base_dir}'tng-100_z0.4_shape'${rdecider}'/shape'${rdecider}'_classes_fe-based_295haloIds.csv'
# classify based on f-e; no masses; no a1a4
if [ $classify_wfe_noa1a4 == 1 ];
then
    outdir=${base_outdir}'classify_wfe_noa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file=${features_file} \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --no_2nd_order_feats --good_radius_feats \
                  --rdecider=${rdecider} --prolate_vs_not
fi
