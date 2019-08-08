#!/bin/bash
shape_datapath='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/tng-100/'
base_outdir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/'
# ------------------------------------------------------------------------------
allclasses_noa1a4=0
twoclasses_noa1a4=0
allclasses_wa1a4=0
twoclasses_wa1a4=0
regress_wT_noa1a4=1
classify_wT_noa1a4=1
regress_wT_wa1a4=0
classify_wT_wa1a4=0
allclasses_noa1a4_wtsne=0
twoclasses_noa1a4_wtsne=0
regress_wT_noa1a4_wtsne=0
classify_wT_noa1a4_wtsne=0
# ------------------------------------------------------------------------------
# run  analysis for one projection;
features_file='/Users/humnaawan/repos/3D-galaxies-kavli/data/tng_highres/xy/features_41.csv'
# no masses included; no a1a4; 3 classes
if [ $allclasses_noa1a4 == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_3classes/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res --plot_feature_dists \
                    --no_2nd_order_feats --good_radius_feats
    outdir=${base_outdir}'rf_tng_highres_we/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats --regress
fi
# ------------------------------------------------------------------------------
# 2 classes; no masses included; no a1a4
if [ $twoclasses_noa1a4 == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_2classes/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats \
                    --prolate_vs_not --plot_feature_dists
fi
# ------------------------------------------------------------------------------
# now include a1a4
# no masses included; 3 classes
if [ $allclasses_wa1a4 == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_3classes_wa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res --plot_feature_dists \
                    --good_radius_feats
    outdir=${base_outdir}'rf_tng_highres_we_wa1a4/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --good_radius_feats --regress
fi
# ------------------------------------------------------------------------------
# 2 classes; no masses included
if [ $twoclasses_wa1a4 == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_2classes_wa1a4/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --good_radius_feats \
                    --prolate_vs_not --plot_feature_dists
fi
# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; no a1a4
if [ $regress_wT_noa1a4 == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_Tonly/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats --regress \
                    --triaxiality_based --rdecider=50
fi
# ------------------------------------------------------------------------------
# classify based on triaxiality; no masses; no a1a4
if [ $classify_wT_noa1a4 == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_2classes_T-based/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats \
                    --triaxiality_based --plot_feature_dists --rdecider=50
fi
# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; with a1a4
if [ $regress_wT_wa1a4 == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_Tonly_wa1a4/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --good_radius_feats --regress \
                    --triaxiality_based
fi
# ------------------------------------------------------------------------------
# classify based on triaxiality; no masses; with a1a4
if [ $classify_wT_wa1a4 == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_2classes_T-based_wa1a4/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --good_radius_feats \
                    --triaxiality_based --plot_feature_dists
fi
# ------------------------------------------------------------------------------
# no masses included; no a1a4; 3 classes; w/ tsne features
if [ $allclasses_noa1a4_wtsne == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_3classes_wtsne/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res --plot_feature_dists \
                    --no_2nd_order_feats --good_radius_feats \
                    --add_tsne_feats --n_comps=3
    outdir=${base_outdir}'rf_tng_highres_we_wtsne/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats --regress \
                    --add_tsne_feats --n_comps=3
fi
# ------------------------------------------------------------------------------
# 2 classes; no masses included; no a1a4; wtsne
if [ $twoclasses_noa1a4_wtsne == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_2classes_wtsne/'
    # classification
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res \
                    --no_2nd_order_feats --good_radius_feats \
                    --prolate_vs_not --add_tsne_feats --n_comps=2 --plot_feature_dists
fi

# ------------------------------------------------------------------------------
# regress with triaxiality; no masses; no a1a4; wtsne
if [ $regress_wT_noa1a4_wtsne == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_Tonly_wtsne/'
    # regression
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res --plot_feature_dists \
                    --no_2nd_order_feats --good_radius_feats --regress \
                    --triaxiality_based --add_tsne_feats --n_comps=2
fi
# ------------------------------------------------------------------------------
# classify based on triaxiality; no masses; no a1a4; wtsne
if [ $classify_wT_noa1a4_wtsne == 1 ];
then
    outdir=${base_outdir}'rf_tng_highres_we_2classes_T-based_wtsne/'
    python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/run_rf.py \
                    --features_file=${features_file} \
                    --shape_datapath=${shape_datapath} \
                    --outdir=${outdir} --high_res --plot_feature_dists \
                    --no_2nd_order_feats --good_radius_feats \
                    --triaxiality_based --add_tsne_feats --n_comps=2
fi
