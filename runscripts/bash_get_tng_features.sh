#!/bin/bash

summary_datapath='/Users/humnaawan/repos/3D-galaxies-kavli/data/tng_highres/xy/'
shape_datapath='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/tng-100/'
outdir='/Users/humnaawan/repos/3D-galaxies-kavli/outputs/extracted_features/'

python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_features.py \
                --summary_datapath=${summary_datapath} \
                --shape_datapath=${shape_datapath} --outdir=${outdir} \
                --data_tag='xy'
python /Users/humnaawan/repos/3D-galaxies-kavli/runscripts/get_features.py \
                --summary_datapath=${summary_datapath} \
                --shape_datapath=${shape_datapath} --outdir=${outdir} \
                --rdecider=50 --data_tag='xy'
