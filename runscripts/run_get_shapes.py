import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import datetime, time, socket, os
import numpy as np, pickle
from d3g2d import get_shape_main, readme as readme_obj, get_time_passed, rcparams
from d3g2d import tng_snap2z, illustris_snap2z, summary_datapath, get_shape_class
for key in rcparams: mpl.rcParams[key] = rcparams[key]
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--data_dir', dest='data_dir',
                  help='Path to the folder with the data.')
parser.add_option('--q',
                  action='store_false', dest='quiet', default=False,
                  help='No print statements.')
parser.add_option('--test',
                  action='store_true', dest='test', default=False,
                  help="Test against Hongys's output.")
parser.add_option('--illustris',
                  action='store_true', dest='illustris', default=False,
                  help="Use the stellar mass data.")
parser.add_option('--just_plots',
                  action='store_true', dest='just_plots', default=False,
                  help="Just plot the figures; assume data is already saved.")
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
data_dir = options.data_dir
quiet = options.quiet
test = options.test
illustris = options.illustris
just_plots = options.just_plots
# ------------------------------------------------------------------------------
start_time0 = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running run_get_shapes on %s\n\n' % socket.gethostname()
update += 'data_dir: %s\n' % data_dir
update += '\nOptions:\n%s\n' % options
readme = readme_obj(outdir=data_dir, readme_tag=readme_tag, first_update=update)
readme.run()

if test or illustris:
    z = illustris_snap2z['z']
    sim_name = 'Illustris-1'
    if test:
        Rstar = np.arange(1, 101, 1)
        haloIDs = [5, 16941]
    else:
        Rstar = np.hstack( [ np.arange(1, 101, 1), np.arange(110, 160, 10) ])
        haloIDs = []
        for f in [f for f in os.listdir( data_dir ) if f.startswith( sim_name )]:
            haloIDs.append( int( f.split('halo')[1].split('_z')[0] ) )
        haloIDs = haloIDs
else:
    z = tng_snap2z['z']
    sim_name = 'TNG100-1'
    haloIDs = []
    for f in [f for f in os.listdir(summary_datapath) if f.startswith( sim_name ) ]:
        haloIDs.append( int(f.split('_')[1]) )
    Rstar = np.arange(20, 160, 10)
shape_tag = 'shape_%sRvals' % len( Rstar )
readme.update(to_write='Working with %s galaxies for %s' % ( len(haloIDs), sim_name))

# run analysis to get axis ratios etc.
if not just_plots:
    for haloID in haloIDs:
        start_time = time.time()
        update = 'Getting shape data for halo %s'  % haloID
        readme.update(to_write=update)
        if not quiet: print(update)
        filename = get_shape_main(source_dir='%s/%s_halo%s_z%s' % (data_dir, sim_name, haloID, z),
                                  fname='cutout_%s.hdf5' % haloID,
                                  illustris=test or illustris, Rstar=Rstar)
        update = 'Saved %s\n' % filename
        update += '## Time taken: %s\n'%get_time_passed(start_time)
        readme.update(to_write=update)
        if not quiet: print(update)
# ------------------------------------------------------------------------------
# now classify
Rdecider = 100
if not just_plots:
    update = 'Getting shape classification ... \n'
    start_time = time.time()
    filename = get_shape_class(data_dir=data_dir, startwith_tag=sim_name,
                               shape_tag=shape_tag, Rdecider=Rdecider)
    update += 'Saved %s\n' % filename
    update += '## Time taken: %s\n'%get_time_passed(start_time)
    readme.update(to_write=update)
    if not quiet: print(update)

# ------------------------------------------------------------------------------
# plot some figs for the shapes: read individual data
# set up the figs directory
fig_dir = '%s/figs/' % data_dir
os.makedirs(fig_dir, exist_ok=True)
# read in the shapes
file = [ f for f in os.listdir(data_dir) if f.startswith('shape%s_' % Rdecider)][0]
with open('%s/%s' % (data_dir, file), 'rb') as f:
    shape_data = pickle.load(f)
colors = {'P': 'r', 'S': 'g', 'T': 'c', 'O': 'b'}
# set up the figures
nrows, ncols = 3, 1
# figure for the actual values
fig1, axes1 = plt.subplots(nrows, ncols)
plt.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
# figure for the differences
fig2, axes2 = plt.subplots(nrows, ncols)
plt.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
# figure for the histograms
fig3, axes3 = plt.subplots(nrows, ncols)
plt.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
bins = np.arange(0, 1, 0.01)
# initiate counters
counter_p, counter_o, counter_s, counter_t = 0, 0, 0, 0
# also store the values for vals for recider
shape_data_vals = {}
# loop over the local foders; each folder is for a specific halo
for i, folder in enumerate([f for f in os.listdir(data_dir) if f.startswith(sim_name)]):
    haloId = int(folder.split('halo')[-1].split('_')[0])
    ind = np.where(np.array( shape_data['haloId'] ) == haloId)[0]
    shape_class = np.array(shape_data['shape'])[ind][0]
    if shape_class == 'P': counter_p += 1
    if shape_class == 'O': counter_o += 1
    if shape_class == 'S': counter_s += 1
    if shape_class == 'T': counter_t += 1
    # ---------------------------------------------------------------------
    # now read in the data produced from my version
    file = [ f for f in os.listdir('%s/%s' % (data_dir, folder)) if f.startswith('shape_')]
    if len(file) == 1:
        file = file[0]
        with open('%s/%s/%s' % (data_dir, folder, file), 'rb') as f:
            data_now = pickle.load(f)
        # calculate triaxiality
        data_now['T'] = (1 -  data_now['b/a'] ** 2 ) / (1 -  data_now['c/a'] ** 2 )
        # figure 1
        axes1[0].plot( data_now['Rstar'], data_now['b/a'], '.-', alpha=0.4, color=colors[shape_class])
        axes1[1].plot( data_now['Rstar'], data_now['c/a'], '.-', alpha=0.4, color=colors[shape_class])

        axes1[2].plot( data_now['Rstar'], data_now['T'], '.-',alpha=0.4, color=colors[shape_class])
        # figure 2
        rmean = ( data_now['Rstar'][:-1] +  data_now['Rstar'][1:]) / 2
        axes2[0].plot( rmean, np.diff( data_now['b/a'] ), '.-',alpha=0.4, color=colors[shape_class])
        axes2[1].plot( rmean, np.diff( data_now['c/a'] ), '.-',alpha=0.4, color=colors[shape_class])
        axes2[2].plot( rmean, np.diff( data_now['T'] ), '.-',alpha=0.4, color=colors[shape_class])
        # figure 3
        # plot b/a
        axes3[0].hist( data_now['b/a'], density=True,
                      histtype='step', lw=2, alpha=0.4, bins=bins, color=colors[shape_class])
        # plot c/a
        axes3[1].hist( data_now['c/a'], density=True,
                      histtype='step', lw=2, alpha=0.4, bins=bins, color=colors[shape_class])
        # plot triaxiality
        axes3[2].hist( data_now['T'],  density=True,
                      histtype='step', lw=2, alpha=0.4, bins=bins, color=colors[shape_class])
        # store values at Rdecider
        ind = np.where( data_now['Rstar'] == Rdecider )[0]
        if i == 0:
            shape_data_vals['b/a_%s' % Rdecider] = [ data_now['b/a'][ind] ]
            shape_data_vals['c/a_%s' % Rdecider] = [ data_now['c/a'][ind] ]
            shape_data_vals['T_%s' % Rdecider] = [ data_now['T'][ind] ]
        else:
            shape_data_vals['b/a_%s' % Rdecider] += [ data_now['b/a'][ind] ]
            shape_data_vals['c/a_%s' % Rdecider] += [ data_now['c/a'][ind] ]
            shape_data_vals['T_%s' % Rdecider] += [ data_now['T'][ind] ]
    else:
        raise ValueError('Somethings wrong: file array: %s' % file)
# finalize plots
# figure 1
axes1[0].set_title('Total: %s galaxies' % i)
axes1[0].set_ylabel('b/a')
axes1[1].set_ylabel('c/a')
axes1[2].set_ylabel('T')
axes1[2].set_xlabel('R (kpc)')
fig1.set_size_inches(15, 15)
# add legend
custom_lines = [Line2D([0], [0], color=colors['P'], lw=10),
                Line2D([0], [0], color=colors['O'], lw=10),
                Line2D([0], [0], color=colors['T'], lw=10),
                Line2D([0], [0], color=colors['S'], lw=10)]
axes1[0].legend(custom_lines,
               ['Prolate (N=%s)' % counter_p,
                'Oblate (N=%s)' % counter_o,
                'Triaxial (N=%s)' % counter_t,
                'Spherical (N=%s)' % counter_s],
               bbox_to_anchor=(1, 0.9), frameon=False)
# save figure
filename = 'shape%s_trends.png' % (Rdecider)
fig1.savefig('%s/%s'%(fig_dir, filename), format='png',
             bbox_inches='tight')
plt.close(fig1)
readme.update(to_write='Saved %s\n' % filename)
# figure 2
axes2[0].set_title('Total: %s galaxies' % i)
axes2[0].set_ylabel(r'$\Delta$ b/a')
axes2[1].set_ylabel(r'$\Delta$ c/a')
axes2[2].set_ylabel(r'$\Delta$ T')
axes2[2].set_xlabel('R (kpc)')
fig2.set_size_inches(15, 15)
# add legend
axes2[0].legend(custom_lines,
               ['Prolate (N=%s)' % counter_p,
                'Oblate (N=%s)' % counter_o,
                'Triaxial (N=%s)' % counter_t,
                'Spherical (N=%s)' % counter_s],
               bbox_to_anchor=(1, 0.9), frameon=True)
# save figure
filename = 'shape%s_trends_differences.png' % (Rdecider)
fig2.savefig('%s/%s'%(fig_dir, filename), format='png',
             bbox_inches='tight')
plt.close(fig2)
readme.update(to_write='Saved %s\n' % filename)
# figure 3
axes3[0].set_xlabel('b/a')
axes3[1].set_xlabel('c/a')
axes3[2].set_xlabel('T')
for i in range(3):
    axes3[i].set_ylabel('Counts')
axes3[0].legend(custom_lines,
               ['Prolate (N=%s)' % counter_p,
                'Oblate (N=%s)' % counter_o,
                'Triaxial (N=%s)' % counter_t,
                'Spherical (N=%s)' % counter_s],
               bbox_to_anchor=(1, 0.9), frameon=False)
fig3.set_size_inches(15, 15)
filename = 'shape%s_histograms.png' % (Rdecider)
fig3.savefig('%s/%s'%(fig_dir, filename), format='png',
             bbox_inches='tight')
plt.close('all')
readme.update(to_write='Saved %s\n' % filename)

# flatten the data
for key in shape_data_vals:
    shape_data_vals[key] = np.array(shape_data_vals[key]).flatten()
# also plot the axis ratios
counters = {}
plt.clf()
for shape in np.unique( shape_data['shape'] ):
    ind = np.where( shape_data['shape'] == shape )[0]
    counters[shape] = len(ind)
    plt.plot(shape_data_vals['b/a_100'][ind], shape_data_vals['c/a_100'][ind], '.',
            color=colors[shape])
# add 1-1 line; also the analog as in Hongyu's Fig 2
x = np.arange(0, 1.4, 0.1)
y = x - 0.2
plt.plot(x, x, 'k-')
plt.plot(x, y, 'k-')
# plot details
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('b/a_%s' % Rdecider)
plt.ylabel('c/a_%s' % Rdecider)
# add legend
plt.legend(custom_lines,
           ['Prolate (N=%s)' % counters['P'],
            'Oblate (N=%s)' % counters['O'],
            'Triaxial (N=%s)' % counters['T'],
            'Spherical (N=%s)' % counters['S']],
           loc='best', frameon=True)
# save plot
filename = 'axis_ratios_classification.png'
plt.savefig('%s/%s'%(fig_dir, filename), format='png',
            bbox_inches='tight')
plt.close('all')
readme.update(to_write='Saved %s\n' % filename)

update = 'Done.\n## Time taken: %s\n'%get_time_passed(start_time0)
readme.update(to_write=update)
