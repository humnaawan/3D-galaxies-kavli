import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import pickle, datetime, time, socket, os
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from d3g2d import readme as readme_obj, get_time_passed, rcparams
for key in rcparams: mpl.rcParams[key] = rcparams[key]

from helpers_summarydata_readin import get_features_highres, get_features_lowres, get_features_highres_summed
from helpers_features_plots import summary_data_plots, plot_hongyus_analog, plot_highd_data
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--summary_datapath', dest='summary_datapath',
                  help='Path to the folder with the summary data.')
parser.add_option('--shape_datapath', dest='shape_datapath',
                  help='Path to the folder with the shape data.')
parser.add_option('--outdir', dest='outdir',
                  help='Path to the folder where to save results.')
parser.add_option('--low_res',
                  action='store_true', dest='low_res', default=False,
                  help='Treat data as low resolution data.')
parser.add_option('--rdecider', dest='Rdecider', default=100,
                  help='Radius to consider the shape at.')
parser.add_option('--data_tag', dest='data_tag',
                  help='data_tag to include in the saved filenames.')
parser.add_option('--summed_data', dest='summed_data',
                  action='store_true', default=False,
                  help='Consider the summary data to have multiple projections and nested data structure.')
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
summary_datapath = options.summary_datapath
shape_datapath = options.shape_datapath
outdir = options.outdir
low_res = options.low_res
if low_res: res_tag = 'low_res'
else: res_tag = 'high_res'
Rdecider = int( options.Rdecider )
data_tag = options.data_tag
if data_tag is None: data_tag = ''
data_tag = '_%s_%s_shape%s' % ( res_tag, data_tag, Rdecider )
summed_data = options.summed_data
# check to ensure outdir exists
os.makedirs(outdir, exist_ok=True)
print(outdir)
# ------------------------------------------------------------------------------
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running get_features.py on %s\n\n' % socket.gethostname()
update += 'Options:\n%s\n' % options
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()
# set up the figs directory
fig_dir = '%s/figs%s/' % (outdir, data_tag)
os.makedirs(fig_dir, exist_ok=True)
# start things up
start_time = time.time()
# ------------------------------------------------------------------------------
# read in shape data to get the haloIds
file = [ f for f in os.listdir(shape_datapath) if \
        f.startswith('shape%s_classes' % Rdecider) and f.__contains__('T-based')][0]
shape_data_T_based = pd.read_csv('%s/%s' % (shape_datapath, file) )

file = [ f for f in os.listdir(shape_datapath) if \
        f.startswith('shape%s_classes' % Rdecider) and f.__contains__('axis-ratios-based')][0]
shape_data_abc_based = pd.read_csv('%s/%s' % (shape_datapath, file) )
# ------------------------------------------------------------------------------
# now read the summary data and assemble features
# follow the same order are haloId in the shape data for easy matching later on
for i, haloId in enumerate(shape_data_T_based['haloId']):
    filename = [f for f in os.listdir(summary_datapath) if f.__contains__('_%s_'%haloId)]
    if len(filename) != 1:
        raise ValueError('Somethings wrong: haloId %s: file array: %s' % ( haloId, file) )
    else:
        filename = filename[0]
    data = np.load('%s/%s' % (summary_datapath, filename))
    if low_res:
        out = get_features_lowres(data_for_halo=data)
    else:
        if summed_data:
            out = get_features_highres_summed(data_for_halo=data)
        else:
            out = get_features_highres(data_for_halo=data)
    if i == 0:
        keys = out.keys()
        feats = list(out.values())
    else:
        feats = np.vstack([feats, list(out.values())])
    # --------------------------------------------------------------------------
# assemble the features as a dataframe
feats = pd.DataFrame(feats, columns=keys)
# save the data
filename = 'features_%s%s.csv' % ( len( feats.keys() ), data_tag )
feats.to_csv('%s/%s' % (outdir, filename), index=False)
readme.update(to_write='Saved %s in %s\n' % (filename, outdir))

# --------------------------------------------------------------------------
# create some plots
# axis-ratios based
shape_class_data = shape_data_abc_based
colors_dict = {'P': 'r', 'O': 'b',  'T': 'c', 'S': 'g'}
class_tag = 'axis-ratios-based'
classtag_to_classname = {'P': 'Prolate', 'S': 'Spherical',
                         'T': 'Triaxial', 'O': 'Oblate'}
update = summary_data_plots(outdir=fig_dir, summary_datapath=summary_datapath,
                            Rdecider=Rdecider,
                            shape_class_data=shape_class_data,
                            colors_dict=colors_dict,
                            classtag_to_classname=classtag_to_classname,
                            class_tag=class_tag, summed_data=summed_data)
readme.update(to_write=update)

shape_class_arr = shape_class_data['shape%s_class' % Rdecider].values
for key in ['logm100', 'logm', 'logm30']:
    logmass = feats[key].values
    update = plot_hongyus_analog(outdir=fig_dir, logmass=logmass,
                                 shape_class_arr=shape_class_arr,
                                 Rdecider=Rdecider, class_tag=class_tag,
                                 mass_tag=key )
    readme.update(to_write=update)

fnames = plot_highd_data(outdir=fig_dir, features=feats.values,
                         targets=shape_class_arr,
                         title='classification based on axis ratios',
                         figlabel='axis-ratios-based_shape%s' % Rdecider)
readme.update(to_write='Saved %s\n' % fnames)


# repeat for T-based stuff
shape_class_data = shape_data_T_based
threshold_T = 0.7
colors_dict = {'P': 'r', 'Not-P': 'b'}
class_tag = 'T-based_thres%s' % threshold_T
classtag_to_classname = {'P': r'Prolate (T$>$%s)' % threshold_T,
                         'Not-P': r'Not-Prolate (T$\leq$%s)' % threshold_T}
update = summary_data_plots(outdir=fig_dir, summary_datapath=summary_datapath,
                            Rdecider=Rdecider,
                            shape_class_data=shape_class_data,
                            colors_dict=colors_dict,
                            classtag_to_classname=classtag_to_classname,
                            class_tag=class_tag, summed_data=summed_data)
readme.update(to_write=update)

shape_class_arr = shape_class_data['shape%s_class' % Rdecider].values
for key in ['logm100', 'logm', 'logm30']:
    logmass = feats[key].values
    update = plot_hongyus_analog(outdir=fig_dir, logmass=logmass,
                                 shape_class_arr=shape_class_arr,
                                 Rdecider=Rdecider, class_tag=class_tag,
                                 mass_tag=key )
    readme.update(to_write=update)

fnames = plot_highd_data(outdir=fig_dir, features=feats.values,
                         targets=shape_class_arr,
                         title='classification based on axis ratios',
                         figlabel='axis-ratios-based_shape%s' % Rdecider)
readme.update(to_write='Saved %s\n' % fnames)

# ----------------------------------------------------------------------
readme.update(to_write='## Time taken: %s\n'%get_time_passed(start_time))
