################################################################################
# This script calculates the 3D shape params for each halo, then classifies
# shapes, and saves the compiled shape data.
################################################################################
import datetime, time, socket, os
import numpy as np, pickle
import pandas as pd
from d3g2d import get_shape_main, readme as readme_obj, get_time_passed
from d3g2d import get_shape_class
from helpers_shape_plots import plot_shape_trends, plot_axis_ratios
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--data_dir', dest='data_dir',
                  help='Path to the folder with the data.')
parser.add_option('--outdir', dest='outdir',
                  help='Path to the folder where to store dat.')
parser.add_option('--test',
                  action='store_true', dest='test', default=False,
                  help="Test against Hongys's output.")
parser.add_option('--illustris',
                  action='store_true', dest='illustris', default=False,
                  help="Use the stellar mass data.")
parser.add_option('--rdecider', dest='Rdecider', default=100,
                  help='Radius to consider the shape at.')
parser.add_option('--z', dest='z',
                  help='Redshift.')
parser.add_option('--extended_rstar',
                  action='store_true', dest='extended_rstar', default=False,
                  help="Consider a lot of Rstar values.")
parser.add_option('--post_process_only',
                  action='store_true', dest='post_process_only', default=False,
                  help="Assume shape data for individual haloes is already saved.")
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
data_dir = options.data_dir
outdir = options.outdir
test = options.test
illustris = options.illustris
Rdecider = int( options.Rdecider )
z = float( options.z )
extended_rstar = options.extended_rstar
post_process_only = options.post_process_only
# make the outdir if it doesn't exist
os.makedirs(outdir, exist_ok=True)
# ------------------------------------------------------------------------------
start_time0 = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running run_get_shapes on %s\n\n' % socket.gethostname()
update += 'data_dir: %s\n' % data_dir
update += '\nOptions:\n%s\n' % options
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()

if test or illustris:
    sim_name = 'Illustris-1'
    if test:
        Rstar = np.arange(1, 101, 1)
        haloIds = [5, 16941]
else:
    sim_name = 'TNG100-1'

if not test:
    if extended_rstar:
        Rstar = np.hstack( [ np.arange(1, 101, 1), np.arange(110, 160, 10) ])
    else:
        Rstar = np.arange(20, 160, 10)
    # read in the haloIds
    haloIds = np.genfromtxt( '%s/haloIds.txt'% data_dir , dtype=int)
# set up shape tag
shape_tag = 'shape_%sRvals' % len( Rstar )
readme.update(to_write='Working with %s galaxies for %s\n' % (len(haloIds), sim_name))

if not post_process_only:
    readme.update(to_write='Getting shape data for all halos ... ')
    # run analysis to get axis ratios etc for each halo
    for haloId in haloIds:
        start_time = time.time()
        update = 'Getting shape data for halo %s'  % haloId
        readme.update(to_write=update)
        filename = get_shape_main(source_dir='%s/%s_halo%s_z%s' % (data_dir, sim_name, haloId, z),
                                  fname='cutout_%s.hdf5' % haloId, z=z,
                                  illustris=test or illustris, Rstar=Rstar)
        update = 'Saved %s\n' % filename
        update += '## Time taken: %s\n'%get_time_passed(start_time)
        readme.update(to_write=update)
# ------------------------------------------------------------------------------
# now save the shape data (b/a, c/a, T) for all the haloes compiled together
# initiate storing dictionaru
shape_data = {}
for key in ['b/a', 'c/a', 'T', 'p', 'q', 'flattening', 'elongation']:
    shape_data[ '%s_%s' % (key, Rdecider) ] = []
shape_data['haloId'] = []
# loop over the haloIds
for haloId in haloIds:
    # looking at the folder for this halo
    # ---------------------------------------------------------------------
    folder = [f for f in os.listdir(data_dir) if f.startswith(sim_name) and f.__contains__( 'halo%s_' % haloId)]
    if len(folder) != 1:
        raise ValueError('Something is wrong: haloId %s: folder list: %s' % (haloId, folder))
    folder = folder[0]
    # file the shape file
    file = [ f for f in os.listdir('%s/%s' % (data_dir, folder)) if f.startswith(shape_tag)]
    if len(file) != 1:
        raise ValueError('Something is wrong: haloId %s: folder: %s: file list: %s' % (haloId, folder, file))
    file = file[0]
    # read in the data
    with open('%s/%s/%s' % (data_dir, folder, file), 'rb') as f:
        data = pickle.load(f)
    # add triaxiality
    data['T'] = (1 -  data['b/a'] ** 2 ) / (1 -  data['c/a'] ** 2 )
    data['p'] = data['c/a'] / data['b/a']
    data['q'] = data['b/a']
    data['flattening'] = np.sqrt( 1 - data['p'] ** 2 )
    data['elongation'] = np.sqrt( 1 - data['q'] ** 2 )
    # find value specified rstar
    ind = np.where( data['Rstar'] == Rdecider )[0]
    # store the values for this Rstar
    shape_data['haloId'] += [ haloId ]
    shape_data['b/a_%s' % Rdecider] += [ data['b/a'][ind] ]
    shape_data['c/a_%s' % Rdecider] += [ data['c/a'][ind] ]
    shape_data['T_%s' % Rdecider] += [ data['T'][ind] ]
    shape_data['p_%s' % Rdecider] += [ data['p'][ind] ]
    shape_data['q_%s' % Rdecider] += [ data['q'][ind] ]
    shape_data['flattening_%s' % Rdecider] += [ data['flattening'][ind] ]
    shape_data['elongation_%s' % Rdecider] += [ data['elongation'][ind] ]
# flatten the data
for key in shape_data:
    shape_data[key] = np.array( shape_data[key] ).flatten()
shape_data = pd.DataFrame( shape_data )
# save the data
filename = 'shape%s_data_%shaloIds.csv' % (Rdecider, len(haloIds) )
shape_data.to_csv('%s/%s' % (outdir, filename), index=False)
readme.update(to_write='Saved %s\n' % filename)

# ------------------------------------------------------------------------------
# now classify
update = 'Getting shape classification based on axis ratios... \n'
filename1 = get_shape_class(outdir=outdir, shape_data_dict=shape_data,
                           axis_ratios_based=True, Rdecider=Rdecider)
readme.update(to_write='Saved %s\n' % filename1)

threshold_T = 0.7
update = 'Getting shape classification based on T... \n'
filename2 = get_shape_class(outdir=outdir, shape_data_dict=shape_data,
                           axis_ratios_based=False, Rdecider=Rdecider,
                           threshold_T=threshold_T)
readme.update(to_write='Saved %s\n' % filename2)

# ------------------------------------------------------------------------------
# plot some things
# ------------------------------------------------------------------------------
fig_dir = '%s/figs/' % outdir
os.makedirs(fig_dir, exist_ok=True)
# first the axis-ratios based data
shape_class_data = pd.read_csv('%s/%s' % (outdir, filename1 ))
colors_dict = {'P': 'r', 'O': 'b',  'T': 'c', 'S': 'g'}
class_tag = 'axis-ratios-based'
classtag_to_classname = {'P': 'Prolate', 'S': 'Spherical',
                         'T': 'Triaxial', 'O': 'Oblate'}
update = plot_shape_trends(outdir=fig_dir, data_dir=data_dir, sim_name=sim_name,
                           shape_tag=shape_tag, Rdecider=Rdecider,
                           shape_class_data=shape_class_data,
                           colors_dict=colors_dict,
                           classtag_to_classname=classtag_to_classname,
                           class_tag=class_tag)
readme.update(to_write=update)

update = plot_axis_ratios(outdir=fig_dir, shape_data=shape_data, Rdecider=Rdecider,
                          shape_class_data=shape_class_data,
                          colors_dict=colors_dict,
                          classtag_to_classname=classtag_to_classname,
                          class_tag=class_tag)
readme.update(to_write=update)

# now the T based data
shape_class_data = pd.read_csv('%s/%s' % (outdir, filename2 ))
colors_dict = {'P': 'r', 'Not-P': 'b'}
class_tag = 'T-based_thres%s' % threshold_T
classtag_to_classname = {'P': r'Prolate (T$>$%s)' % threshold_T,
                         'Not-P': r'Not-Prolate (T$\leq$%s)' % threshold_T}
update = plot_shape_trends(outdir=fig_dir, data_dir=data_dir, sim_name=sim_name,
                           shape_tag=shape_tag, Rdecider=Rdecider,
                           shape_class_data=shape_class_data,
                           colors_dict=colors_dict,
                           classtag_to_classname=classtag_to_classname,
                           class_tag=class_tag)
readme.update(to_write=update)

update = plot_axis_ratios(outdir=fig_dir, shape_data=shape_data, Rdecider=Rdecider,
                          shape_class_data=shape_class_data,
                          colors_dict=colors_dict,
                          classtag_to_classname=classtag_to_classname,
                          class_tag=class_tag)
readme.update(to_write=update)
# ------------------------------------------------------------------------------
# done
readme.update(to_write='Done.\n## Time taken: %s\n' % get_time_passed(start_time0))
