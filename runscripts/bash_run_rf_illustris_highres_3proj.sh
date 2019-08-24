#!/bin/bash

base_dir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/'
# ------------------------------------------------------------------------------
regress_wT_noa1a4=0
classify_wT_noa1a4=0
regress_wT_noa1a4_wmass=0
classify_wT_noa1a4_wmass=0
regress_wT_wa1a4=0
classify_wT_wa1a4=0

classify_wfe_noa1a4=1
classify_wfe_noa1a4_wmass=1
classify_wfe_wa1a4=1
# ------------------------------------------------------------------------------
# run  analysis for all three projections
rdecider=50
# ------------------------------------------------------------------------------
features_path=${base_dir}'illustris_z0.4_3proj_shape'${rdecider}'_features/'
base_outdir=${base_dir}'illustris_z0.4_shape'${rdecider}'_rf_3proj_'

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# no masses included; no a1a4;
# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; no a1a4
shape_file=${base_dir}'illustris_z0.4_shape'${rdecider}'/shape'${rdecider}'_data_449haloIds.csv'
if [ $regress_wT_noa1a4 == 1 ];
then
    outdir=${base_outdir}'regress_wT_noa1a4/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file='' \
                    --shape_file=${shape_file} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats --regress \
                    --rdecider=${rdecider} --just_triaxiality \
                    --combine_projs --features_path=${features_path}
fi
# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; no a1a4
shape_file=${base_dir}'illustris_z0.4_shape'${rdecider}'/shape'${rdecider}'_classes_T-based_0.7thres_449haloIds.csv'
if [ $classify_wT_noa1a4 == 1 ];
then
    outdir=${base_outdir}'classify_wT_noa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file='' \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --no_2nd_order_feats --good_radius_feats \
                  --rdecider=${rdecider} \
                  --combine_projs --features_path=${features_path}
fi

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# with masses included; no a1a4; 3 classes
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# no masses included; no a1a4;
# ------------------------------------------------------------------------------
# regress with triaxiality; with masses; no a1a4
shape_file=${base_dir}'illustris_z0.4_shape'${rdecider}'/shape'${rdecider}'_data_449haloIds.csv'
if [ $regress_wT_noa1a4_wmass == 1 ];
then
    outdir=${base_outdir}'regress_wT_noa1a4_wmass/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file='' \
                    --shape_file=${shape_file} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats --regress \
                    --rdecider=${rdecider} --just_triaxiality \
                    --combine_projs --features_path=${features_path} --wmasses
fi
# ------------------------------------------------------------------------------
# regress with triaxiality; with masses; no a1a4
shape_file=${base_dir}'illustris_z0.4_shape'${rdecider}'/shape'${rdecider}'_classes_T-based_0.7thres_449haloIds.csv'
if [ $classify_wT_noa1a4_wmass == 1 ];
then
    outdir=${base_outdir}'classify_wT_noa1a4_wmass/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file='' \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --no_2nd_order_feats --good_radius_feats \
                  --rdecider=${rdecider} \
                  --combine_projs --features_path=${features_path} --wmasses
fi

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# no masses included; with a1a4;
# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; with a1a4
shape_file=${base_dir}'illustris_z0.4_shape'${rdecider}'/shape'${rdecider}'_data_449haloIds.csv'
if [ $regress_wT_wa1a4 == 1 ];
then
    outdir=${base_outdir}'regress_wT_wa1a4/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file='' \
                    --shape_file=${shape_file} \
                    --outdir=${outdir} --high_res \
                    --good_radius_feats --regress \
                    --rdecider=${rdecider} --just_triaxiality \
                    --combine_projs --features_path=${features_path}
fi
# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; with a1a4
shape_file=${base_dir}'illustris_z0.4_shape'${rdecider}'/shape'${rdecider}'_classes_T-based_0.7thres_449haloIds.csv'
if [ $classify_wT_wa1a4 == 1 ];
then
    outdir=${base_outdir}'classify_wT_wa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file='' \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --good_radius_feats \
                  --rdecider=${rdecider} \
                  --combine_projs --features_path=${features_path}
fi


#
# ------------------------------------------------------------------------------
# classify with fe
shape_file=${base_dir}'illustris_z0.4_shape'${rdecider}'/shape'${rdecider}'_classes_fe-based_449haloIds.csv'
if [ $classify_wfe_noa1a4 == 1 ];
then
    outdir=${base_outdir}'classify_wfe_noa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file='' \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --good_radius_feats --prolate_vs_not \
                  --rdecider=${rdecider} \
                  --combine_projs --features_path=${features_path} \
                  --no_2nd_order_feats
fi

if [ $classify_wfe_wa1a4 == 1 ];
then
    outdir=${base_outdir}'classify_wfe_wa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file='' \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --good_radius_feats --prolate_vs_not \
                  --rdecider=${rdecider} \
                  --combine_projs --features_path=${features_path}
fi

if [ $classify_wfe_noa1a4_wmass == 1 ];
then
    outdir=${base_outdir}'classify_wfe_noa1a4_wmass/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                  --features_file='' \
                  --shape_file=${shape_file} \
                  --outdir=${outdir} --high_res --plot_feature_dists \
                  --good_radius_feats --prolate_vs_not \
                  --rdecider=${rdecider} \
                  --combine_projs --features_path=${features_path} \
                  --no_2nd_order_feats --wmasses
fi
