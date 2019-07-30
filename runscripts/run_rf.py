import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import datetime, time, socket, os
from d3g2d import run_rf, readme as readme_obj, get_time_passed
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--features_file', dest='features_file',
                  help='Path to the folder with the summary data.')
parser.add_option('--shape_datapath', dest='shape_datapath',
                  help='Path to the folder with the shape data.')
parser.add_option('--outdir', dest='outdir',
                  help='Path to the folder where to save results.')
parser.add_option('--q',
                  action='store_false', dest='quiet', default=False,
                  help='No print statements.')
parser.add_option('--regress',
                  action='store_true', dest='regress', default=False,
                  help='Run regression analysis.')
parser.add_option('--m100',
                  action='store_true', dest='m100', default=False,
                  help='Include m100 in features.')
parser.add_option('--m200',
                  action='store_true', dest='m200', default=False,
                  help='Include m200 in features.')
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
features_file = options.features_file
shape_datapath = options.shape_datapath
outdir = options.outdir
quiet = options.quiet
regress = options.regress
m100 = options.m100
m200 = options.m200
# ------------------------------------------------------------------------------
start_time = time.time()
# make the outdir if it doesn't exist
os.makedirs(outdir, exist_ok=True)
#
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running run_rf.py on %s\n\n' % socket.gethostname()
update += 'Options:\n%s\n' % options
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()

# start things up
start_time = time.time()
# read in the features
feats = pd.read_csv(features_file)
if not m100:
    _ = feats.pop('logm100')
if not m200:
    _ = feats.pop('logm200')

# ------------------------------------------------------------------------------
# read in shape data
if regress:
    Rstar = 100
    shape_data = {}
    # loop over the local foders; each folder is for a specific halo
    for i, folder in enumerate([f for f in os.listdir(shape_datapath) if f.startswith('TNG')]):
        # ---------------------------------------------------------------------
        # now read in the data
        file = [ f for f in os.listdir('%s/%s' % (shape_datapath, folder)) if f.startswith('shape_')][0]
        with open('%s/%s/%s' % (shape_datapath, folder, file), 'rb') as f:
            data_now = pickle.load(f)

        data_now['T'] = (1 -  data_now['b/a'] ** 2 ) / (1 -  data_now['c/a'] ** 2 )

        ind = np.where( data_now['Rstar'] == Rstar )[0]

        if i == 0:
            shape_data['b/a_%s' % Rstar] = [ data_now['b/a'][ind] ]
            shape_data['c/a_%s' % Rstar] = [ data_now['c/a'][ind] ]
            shape_data['T_%s' % Rstar] = [ data_now['T'][ind] ]
        else:
            shape_data['b/a_%s' % Rstar] += [ data_now['b/a'][ind] ]
            shape_data['c/a_%s' % Rstar] += [ data_now['c/a'][ind] ]
            shape_data['T_%s' % Rstar] += [ data_now['T'][ind] ]

    for key in shape_data:
        shape_data[key] = np.array(shape_data[key]).flatten()
    shape_data = pd.DataFrame( shape_data )
else:
    file = [ f for f in os.listdir(shape_datapath) if f.startswith('shape100_')][0]
    with open('%s/%s' % (shape_datapath, file), 'rb') as f:
        shape = pickle.load(f)
    shape_data = pd.DataFrame({ 'shape': np.array( shape['shape'] ) } )
    # combine spherical vs triaxial
    shape_data['shape'][ shape_data['shape'] == 'S' ] = 'T'

# update
update = '## Running analysis with %s features:\n%s\n' % ( len(feats.keys()), feats.keys() )
update += '## For %s targets:\n%s\n' % ( len(shape_data.keys()), shape_data.keys() )
update += '## For %s galaxies\n' % ( np.shape(feats.values)[0] )
readme.update(to_write=update)

# ------------------------------------------------------------------------------
# run random forest
run_rf(feats=feats.values, feat_labels=feats.keys(),
       targets=shape_data.values, target_labels=shape_data.keys(),
       outdir=outdir, regression=regress, readme=readme)

update = '\n## Time taken: %s'%get_time_passed(start_time)
readme.update(to_write=update)
if not quiet: print(update)
