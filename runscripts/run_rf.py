import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import datetime, time, socket, os
from d3g2d import run_rf, readme as readme_obj, get_time_passed, rcparams
for key in rcparams: mpl.rcParams[key] = rcparams[key]
from sklearn.manifold import TSNE
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
parser.add_option('--prolate_vs_not',
                  action='store_true', dest='prolate_vs_not', default=False,
                  help='Consider only two classes.')
parser.add_option('--high_res',
                  action='store_true', dest='high_res', default=False,
                  help='Treat data as high resolution data.')
parser.add_option('--combine_projs',
                  action='store_true', dest='combine_projs', default=False,
                  help='Combine features from various projections. \
                  features_file should now be just name of csv file to look for.')
parser.add_option('--features_path', dest='features_path',
                  help='Path to the folder with the summary data for different projections.')
parser.add_option('--no_2nd_order_feats',
                  action='store_true', dest='no_2nd_order_feats', default=False,
                  help='Drop the 2nd-order features.')
parser.add_option('--good_radius_feats',
                  action='store_true', dest='good_radius_feats', default=False,
                  help='Drop any features for below 10kpc and above 100kpc.')
parser.add_option('--plot_feature_dists',
                  action='store_true', dest='plot_feature_dists', default=False,
                  help='Plot distribution for the features.')
parser.add_option('--triaxiality_based',
                  action='store_true', dest='triaxiality_based', default=False,
                  help='Using only triaxiality; only valid for regression.')
parser.add_option('--add_tsne_feats',
                  action='store_true', dest='add_tsne_feats', default=False,
                  help='Add tsne probs as features.')
parser.add_option('--n_comps', dest='n_comps', default=2,
                  help='Number of components.')
parser.add_option('--rdecider', dest='Rdecider', default=100,
                  help='Radius to consider the shape at.')
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
features_file = options.features_file
shape_datapath = options.shape_datapath
outdir = options.outdir
quiet = options.quiet
regress = options.regress
m100 = options.m100
m200 = options.m200
prolate_vs_not = options.prolate_vs_not
if prolate_vs_not and regress:
    raise ValueError('prolate_vs_not tag is not valid for regression.')
high_res = options.high_res
if high_res and m200:
    raise ValueError('Dont have logm200 for high res data.')
combine_projs = options.combine_projs
if combine_projs and features_file.__contains__('/'):
    raise ValueError('features_file must be just the name of csv file when combine_projs is used; not a path.')
features_path = options.features_path
no_2nd_order_feats = options.no_2nd_order_feats
good_radius_feats = options.good_radius_feats
plot_feature_dists = options.plot_feature_dists
if plot_feature_dists and regress:
    raise ValueError('plot_feature_dists tag is not valid for regression.')
triaxiality_based = options.triaxiality_based
add_tsne_feats = options.add_tsne_feats
n_comps = int( options.n_comps )
Rdecider = int( options.Rdecider )
# ------------------------------------------------------------------------------
start_time0 = time.time()
# make the outdir if it doesn't exist
os.makedirs(outdir, exist_ok=True)
#
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running run_rf.py on %s\n\n' % socket.gethostname()
update += 'Options:\n%s\n' % options
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()

if high_res and m100:
    readme.update(to_write='\n## m100 is not exactly m100.')
# start things up
start_time = time.time()
# ------------------------------------------------------------------------------
# read in shape data
if regress or triaxiality_based:
    shape_data = {}
    # loop over the local foders; each folder is for a specific halo
    for i, folder in enumerate([f for f in os.listdir(shape_datapath) if f.startswith('TNG')]):
        # ---------------------------------------------------------------------
        # now read in the data
        file = [ f for f in os.listdir('%s/%s' % (shape_datapath, folder)) if f.startswith('shape_')][0]
        with open('%s/%s/%s' % (shape_datapath, folder, file), 'rb') as f:
            data_now = pickle.load(f)
        # add triaxiality
        data_now['T'] = (1 -  data_now['b/a'] ** 2 ) / (1 -  data_now['c/a'] ** 2 )
        # find value specified rstar
        ind = np.where( data_now['Rstar'] == Rdecider )[0]
        if i == 0:
            if not triaxiality_based:
                shape_data['b/a_%s' % Rdecider] = [ data_now['b/a'][ind] ]
                shape_data['c/a_%s' % Rdecider] = [ data_now['c/a'][ind] ]
            shape_data['T_%s' % Rdecider] = [ data_now['T'][ind] ]
        else:
            if not triaxiality_based:
                shape_data['b/a_%s' % Rdecider] += [ data_now['b/a'][ind] ]
                shape_data['c/a_%s' % Rdecider] += [ data_now['c/a'][ind] ]
            shape_data['T_%s' % Rdecider] += [ data_now['T'][ind] ]
    for key in shape_data:
        shape_data[key] = np.array(shape_data[key]).flatten()
    # asseble classification based on triaxiality
    if not regress:
        shape_class = {}
        shape_class['shape'] = np.empty_like( shape_data['T_%s' % Rdecider] ).astype(str)
        shape_class['shape'][:] = 'Not-P'
        shape_class['shape'][ shape_data['T_%s' % Rdecider] > 0.7] = 'P'
        shape_data = shape_class
    shape_data = pd.DataFrame( shape_data )
else:
    file = [ f for f in os.listdir(shape_datapath) if f.startswith('shape%s_' % Rdecider)][0]
    with open('%s/%s' % (shape_datapath, file), 'rb') as f:
        shape = pickle.load(f)
    shape_data = pd.DataFrame({ 'shape': np.array( shape['shape'] ) } )
    # combine spherical vs triaxial
    shape_data['shape'][ shape_data['shape'] == 'S' ] = 'T'
    if prolate_vs_not:
        shape_data['shape'][ shape_data['shape'] != 'P' ] = 'Not-P'

# ------------------------------------------------------------------------------
# read in the features
if combine_projs:
    if plot_feature_dists: feats_proj = {}
    for i, proj_tag in enumerate( ['xy', 'yz'] ):
        readme.update(to_write='Reading in %s features ... ' % proj_tag)
        if i == 0:
            feats_here = pd.read_csv('%s/%s/%s' % (features_path, proj_tag, features_file) )
            feat_columns = feats_here.keys()
            feats = feats_here.values
            shape_data_interm = shape_data.values
        else:
            feats_here = pd.read_csv('%s/%s/%s' % (features_path, proj_tag, features_file) )
            feats = np.vstack( [feats, feats_here.values ] )
            shape_data_interm = np.vstack( [ shape_data_interm, shape_data.values ] )
        # store for plotting
        if plot_feature_dists:
            feats_proj[ proj_tag ] = feats_here
    feats = pd.DataFrame(feats, columns=feat_columns)
    if plot_feature_dists: shape_base = shape_data.copy()
    shape_data = pd.DataFrame(shape_data_interm, columns=shape_data.keys())
else:
    feats = pd.read_csv(features_file)

# ------------------------------------------------------------------------------
# feature clean up; optional
if not m100 and 'logm100' in feats.keys():
    _ = feats.pop('logm100')
if not m200 and 'logm200' in feats.keys():
    _ = feats.pop('logm200')

if no_2nd_order_feats:
    for key in [f for f in feats.keys() if f.startswith('a1') or f.startswith('a4') ]:
        if key in feats:
            _ = feats.pop(key)
if good_radius_feats:
    for key in ['delM_107_150', 'dele_111_160', 'delpa_111_160']:
        if key in feats:
            _ = feats.pop(key)

if add_tsne_feats:
    readme.update(to_write='Adding tSNE probs for %s components' % n_comps)
    tsne = TSNE(n_components=n_comps, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(feats.values)
    for i in range(n_comps):
        feats['tsne-2d-%scomp' % (i+1)] = tsne_results[:,i]
# update
update = '## Running analysis with %s features:\n%s\n' % ( len(feats.keys()), feats.keys() )
update += '## For %s targets:\n%s\n' % ( len(shape_data.keys()), shape_data.keys() )
update += '## For %s galaxies\n' % ( np.shape(feats.values)[0] )
readme.update(to_write=update)
# ------------------------------------------------------------------------------
if plot_feature_dists:
    fig_dir = '%s/figs_features/' % outdir
    os.makedirs(fig_dir, exist_ok=True)
    nbins = 25
    if combine_projs:
        # plot overall + projections
        for col in [f for f in feats.keys() if not f.__contains__('id') and \
                    not f.__contains__('rpix') and \
                    not f.__contains__('rkpc')]:
            data = feats[col].values
            if key.__contains__('maper') or key.__contains__('mu_'):
                data = np.log10( data )
                col = 'log10%s' % col
            min_val, max_val = min( data ), max( data )
            del_val = ( max_val - min_val ) / nbins
            bins = m_arr = np.arange( min_val - del_val, max_val + 2 * del_val, del_val)
            # plot setup
            plt.clf()
            nrows, ncols = len( feats_proj.keys() ) + 1, 1
            fig, axes = plt.subplots(nrows, ncols)
            plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9)
            # loop over classes
            for shape_class in np.unique( shape_data['shape'].values ):
                ind = np.where( shape_data['shape'].values == shape_class )[0]
                axes[0].hist( data[ind], bins=bins, lw=2, histtype='step', label=shape_class )
                # now consider each projection
                for i, proj in enumerate( feats_proj.keys() ):
                    if col in feats_proj[proj]:
                        ind = np.where( shape_base['shape'].values == shape_class )[0]
                        data_proj = feats_proj[proj][col].values
                        if key.__contains__('maper') or key.__contains__('mu_'):
                            data_proj = np.log10( data_proj )
                        axes[i+1].hist( data_proj[ind], bins=bins, lw=2, histtype='step', label=shape_class )

            axes[-1].set_xlabel(col)
            for row in range( nrows ):
                axes[row].set_ylabel('Counts')
                axes[row].legend( bbox_to_anchor=(1,1) )
            axes[0].set_title( 'all' )
            for i, proj in enumerate( feats_proj.keys() ):
                axes[i+1].set_title( proj )
            fig.set_size_inches(5 * nrows, 5 * nrows)
            filename = 'plot_dist_%s.png' % (col)
            # save file
            plt.savefig('%s/%s'%(fig_dir, filename), format='png',
                        bbox_inches='tight')
            plt.close('all')
            readme.update(to_write='Saved %s' % filename)
    else:
        # plot
        for col in [f for f in feats.keys() if not f.__contains__('id') and \
                    not f.__contains__('rpix') and \
                    not f.__contains__('rkpc')]:
            data = feats[col].values
            if key.__contains__('maper') or key.__contains__('mu_'):
                data = np.log10( data )
                col = 'log10%s' % col
            min_val, max_val = min( data ), max( data )
            del_val = ( max_val - min_val ) / nbins
            bins = m_arr = np.arange( min_val - del_val, max_val + 2 * del_val, del_val)
            plt.clf()
            for shape_class in np.unique( shape_data['shape'].values ):
                ind = np.where( shape_data['shape'].values == shape_class )[0]
                plt.hist( data[ind], bins=bins, lw=2, histtype='step', label=shape_class )
            plt.xlabel(col)
            plt.ylabel('Counts')
            plt.legend( bbox_to_anchor=(1,1) )
            filename = 'plot_dist_%s.png' % (col)
            # save file
            plt.savefig('%s/%s'%(fig_dir, filename), format='png',
                        bbox_inches='tight')
            plt.close('all')
            readme.update(to_write='Saved %s' % filename)
# ------------------------------------------------------------------------------
# run random forest
run_rf(feats=feats.values, feat_labels=feats.keys(),
       targets=shape_data.values, target_labels=shape_data.keys(),
       outdir=outdir, regression=regress, readme=readme)
update = '\n## Time taken: %s'%get_time_passed(start_time)
readme.update(to_write=update)
if not quiet: print(update)

update = 'Done.\n## Time taken: %s\n'%get_time_passed(start_time0)
readme.update(to_write=update)
