import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import pickle, datetime, time, socket, os
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from d3g2d import run_rf, readme as readme_obj, get_time_passed, rcparams
for key in rcparams: mpl.rcParams[key] = rcparams[key]

from helpers_data_readin import get_features_highres, get_features_lowres
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
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
summary_datapath = options.summary_datapath
shape_datapath = options.shape_datapath
outdir = options.outdir
low_res = options.low_res
Rdecider = int( options.Rdecider )
# ------------------------------------------------------------------------------
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running get_features.py on %s\n\n' % socket.gethostname()
update += 'Options:\n%s\n' % options
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()
# set up the figs directory
fig_dir = '%s/figs/' % outdir
os.makedirs(fig_dir, exist_ok=True)
# start things up
start_time = time.time()
# ------------------------------------------------------------------------------
# read in shape data to get the haloIds
file = [ f for f in os.listdir(shape_datapath) if f.startswith('shape%s_' % Rdecider)][0]
with open('%s/%s' % (shape_datapath, file), 'rb') as f:
    shape_data = pickle.load(f)
# ------------------------------------------------------------------------------
# now read the summary data and assemble features
# follow the same order are haloId in the shape data for easy matching later on
for i, haloId in enumerate(shape_data['haloId']):
    filename = [f for f in os.listdir(summary_datapath) if f.__contains__('_%s_'%haloId)]
    if len(filename) != 1:
        print(filename, haloId)
        break
    else:
        filename = filename[0]
    data = np.load('%s/%s' % (summary_datapath, filename))
    if low_res:
        out = get_features_lowres(data_for_halo=data)
    else:
        out = get_features_highres(data_for_halo=data)
    if i == 0:
        keys = out.keys()
        feats = list(out.values())
    else:
        feats = np.vstack([feats, list(out.values())])

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # plot some things for just one galaxy
    if i == 10:
        # ----------------------------------------------------------------------
        # plot the mass porfiles
        plt.clf()
        if low_res:
            plt.plot(data['rpix_shape'] * 5.333, data['mu_shape'], 'o-', label='mu_shape')
            plt.plot(data['aper_rkpc'], 10 ** data['aper_logms'], 'o-', label='aper_logms')
            plt.plot(data['rpix_shape'] * 5.333, data['maper_shape'], 'o-', label='maper_shape')
            plt.plot(data['rpix_prof'] * 5.333, data['maper_prof'], 'x-', label='maper_prof')
            plt.plot(data['rpix_prof'] * 5.333, data['mu_prof'], 'x-', label='mu_prof')
        else:
            plt.plot(data['rkpc_shape'], data['mu_shape'], 'o-', label='mu_shape')
            plt.plot(data['aper_rkpc'], 10 ** data['aper_logms'], 'o-', label='aper_logms')
            plt.plot(data['rkpc_shape'], data['maper_shape'], 'o-', label='maper_shape')
            plt.plot(data['rkpc_prof'], data['maper_prof'], 'x-', label='maper_prof')
            plt.plot(data['rkpc_prof'], data['mu_prof'], 'x-', label='mu_prof')

        # plot details
        plt.legend(bbox_to_anchor=(1,1))
        plt.xlabel('kpc')
        plt.ylabel(r'$\mu$')
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        # save plot
        filename = 'mass_profile_halo%s.png' % haloId
        plt.savefig('%s/%s'%(fig_dir, filename), format='png',
                    bbox_inches='tight')
        plt.close('all')
        readme.update(to_write='Saved %s\n' % filename)
    # --------------------------------------------------------------------------
# assemble the features as a dataframe
feats = pd.DataFrame(feats, columns=keys)
# save the data
filename = 'features_%s.csv' % len( feats.keys() )
feats.to_csv('%s/%s' % (outdir, filename), index=False)
readme.update(to_write='Saved %s in %s\n' % (filename, outdir))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# plot some more plots
# ----------------------------------------------------------------------
# plot all the mass profiles and color-code with shape
colors = {'P': 'r', 'O': 'b', 'S': 'k', 'T': 'c'}
counters = {}
for key in colors: counters[key] = 0

plt.clf()
for i, haloId in enumerate(shape_data['haloId']):
    filename = [f for f in os.listdir(summary_datapath) if f.__contains__('_%s_'%haloId)]
    if len(filename) != 1:
        print(filename, haloId)
        break
    else:
        filename = filename[0]
    data = np.load('%s/%s' % (summary_datapath, filename))
    # find the shape
    shape_class = np.array(shape_data['shape'])[i][0]
    # add to the right counter
    counters[shape_class] += 1
    # plot
    if low_res:
        plt.plot( data['rpix_prof'] * 5.333, np.log10(data['mu_prof']), '.-', alpha=0.3, color=colors[shape_class])
    else:
        plt.plot( data['rkpc_prof'], np.log10(data['mu_prof']), '.-', alpha=0.3, color=colors[shape_class])
# plot details
plt.xlabel('R (kpc)' )
plt.ylabel(r'log10($\mu_{prof}$) ')
plt.xlim(50, 1000)
plt.ylim(0, 10)
# add legend
custom_lines = [Line2D([0], [0], color=colors['P'], lw=10),
                Line2D([0], [0], color=colors['O'], lw=10),
                Line2D([0], [0], color=colors['T'], lw=10),
                Line2D([0], [0], color=colors['S'], lw=10)]
plt.legend(custom_lines,
           ['Prolate (%s)' % counters['P'],
            'Oblate (%s)' % counters['O'],
            'Triaxial (%s)' % counters['T'],
            'Spherical (%s)' % counters['S']],
           bbox_to_anchor=(1, 1), frameon=True)
# save plot
filename = 'mass_profiles_%sgals_shape%s.png' % (i+1, Rdecider)
plt.savefig('%s/%s'%(fig_dir, filename), format='png',
            bbox_inches='tight')
plt.close('all')
readme.update(to_write='Saved %s\n' % filename)
# ----------------------------------------------------------------------
# now redo the plot above + include subplots for various quartiles in logM100
for key in colors: counters[key] = 0

plt.clf()
nrows, ncols = 5, 1
fig, axes = plt.subplots(nrows, ncols)
plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9)

logm100s = feats['logm100'].values
counter_1, counter_2, counter_3, counter_4 = 0, 0, 0, 0
for i, haloId in enumerate(shape_data['haloId']):
    filename = [f for f in os.listdir(summary_datapath) if f.__contains__('_%s_'%haloId)][0]
    data = np.load('%s/%s' % (summary_datapath, filename))

    # find the shape
    shape_class = np.array(shape_data['shape'])[i][0]
    # add to the right counter
    counters[shape_class] += 1
    # plot; all masses
    if low_res:
        axes[0].plot( data['rpix_prof'] * 5.333, np.log10(data['mu_prof']), '.-', alpha=0.3, color=colors[shape_class])
    else:
        axes[0].plot( data['rkpc_prof'], np.log10(data['mu_prof']), '.-', alpha=0.3, color=colors[shape_class])

    # now determine where the logM100 of this galaxies lies in the overall distribution.
    # consider each quartile separately
    q1, q2 = 0, 0.25
    if ( logm100s[i] >= np.quantile(logm100s, q1)) and ( logm100s[i] < np.quantile(logm100s, q2) ):
        axis_ind = 1
        counter_1 += 1
    q1, q2 = 0.25, 0.5
    if ( logm100s[i] >= np.quantile(logm100s, q1)) and ( logm100s[i] < np.quantile(logm100s, q2) ):
        axis_ind = 2
        counter_2 += 1
    q1, q2 = 0.5, 0.75
    if ( logm100s[i] >= np.quantile(logm100s, q1)) and ( logm100s[i] < np.quantile(logm100s, q2) ):
        axis_ind = 3
        counter_3 += 1
    q1, q2 = 0.75, 1.0
    if ( logm100s[i] >= np.quantile(logm100s, q1)) and ( logm100s[i] <= np.quantile(logm100s, q2) ):
        axis_ind = 4
        counter_4 += 1
    # now plot
    if low_res:
        axes[axis_ind].plot( data['rpix_prof'] * 5.333, np.log10(data['mu_prof']), '.-',
                            alpha=0.3, color=colors[shape_class])
    else:
        axes[axis_ind].plot( data['rkpc_prof'], np.log10(data['mu_prof']), '.-',
                            alpha=0.3, color=colors[shape_class])
# plot details
fig.set_size_inches(15, 15)
axes[4].set_xlabel('R (kpc)' )
for row in range( nrows ):
    axes[row].set_ylabel(r'log10($\mu_{\mathrm{prof}}$) ')
    axes[row].set_xlim(50, 1000)
    axes[row].set_ylim(0, 10)
# titles
axes[0].set_title('all masses (%s galaxies)' % ( counter_1 + counter_2 + counter_3 + counter_4))
q1, q2 = 0, 0.25
axes[1].set_title(r' %.2f $\leq$ logM100 < %.2f (%s galaxies)' % (np.quantile(logm100s, q1),
                                                                  np.quantile(logm100s, q2),
                                                                  counter_1))
q1, q2 = 0.25, 0.5
axes[2].set_title(r' %.2f $\leq$ logM100 < %.2f (%s galaxies)' % (np.quantile(logm100s, q1),
                                                                  np.quantile(logm100s, q2),
                                                                  counter_2))
q1, q2 = 0.5, 0.75
axes[3].set_title(r' %.2f $\leq$ logM100 < %.2f (%s galaxies)' % (np.quantile(logm100s, q1),
                                                                  np.quantile(logm100s, q2),
                                                                  counter_3))
q1, q2 = 0.75, 1.0
axes[4].set_title(r' %.2f $\leq$ logM100 $\leq$ %.2f (%s galaxies)' % (np.quantile(logm100s, q1),
                                                                  np.quantile(logm100s, q2),
                                                                  counter_4))
# add legend
axes[0].legend(custom_lines,
               ['Prolate (%s)' % counters['P'],
                'Oblate (%s)' % counters['O'],
                'Triaxial (%s)' % counters['T'],
                'Spherical (%s)' % counters['S']],
               bbox_to_anchor=(1, 1), frameon=True)
# save plot
filename = 'mass_profiles_%sgals_wm100quartiles_shape%s.png' % (i+1, Rdecider)
plt.savefig('%s/%s'%(fig_dir, filename), format='png',
            bbox_inches='tight')
plt.close('all')
readme.update(to_write='Saved %s\n' % filename)
# ----------------------------------------------------------------------
# plot the analog of Hongyu's Fig 3.
for key in ['logm100', 'logm', 'logm30']:
    logmass = feats[key].values
    m_arr = np.arange( min(logmass) - 0.1, max(logmass) + 0.1, 0.1 )
    # set up
    fig, ax1 = plt.subplots()
    # plot normed densities
    ind1 = np.where( shape_data['shape'] == 'P' )[0]
    ax1.hist(logmass[ind1], histtype='step', bins=m_arr, color='r',
             lw=2, density=True )
    ind2 = np.where( shape_data['shape'] != 'P')[0]
    ax1.hist(logmass[ind2], histtype='step', bins=m_arr, color='b',
             lw=2, density=True )
    ax1.set_xlabel( key )
    ax1.set_ylabel( 'Normalized Number Densities' )
    # set up the second axis
    ax2 = ax1.twinx()
    # plot fraction
    frac, ms = [], []
    for j in range(len(m_arr) - 1):
        mlow, mupp = m_arr[j], m_arr[j+1]
        ind = np.where( ( logmass >= mlow ) & ( logmass < mupp ))[0]
        if len(ind) > 0:
            frac.append( len( set(ind1) & set(ind) ) / len( ind) )
            ms.append( np.median( [mlow, mupp ]) )
    ax2.plot(ms, frac, 'k.-', label='Prolate Fraction')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel( 'Prolate Fraction', rotation=-90, labelpad=20)
    ax2.grid(None)
    # details
    custom_lines = [Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='k', lw=2)]
    plt.legend(custom_lines,
               ['Prolate normed density (N=%s)' % len(ind1),
                'Not-Prolate normed density (N=%s)' % len(ind2),
                'Prolate Fraction'],
               bbox_to_anchor=(1.6,1), frameon=True)
    # save plot
    filename = 'hongyu_analog_%sgals_%s_shape%s.png' % (i+1, key, Rdecider)
    plt.savefig('%s/%s'%(fig_dir, filename), format='png',
                bbox_inches='tight')
    plt.close('all')
    readme.update(to_write='Saved %s\n' % filename)
# ----------------------------------------------------------------------
# plot tSNE plots
def plot_highd_data(features, targets, title, figlabel):
    # will look for P  in targets
    n_components = 2
    # get results
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    # set up
    hue = np.array( len( targets ) * ['r'] )
    hue[ targets != 'P' ] = 'g'
    #
    plt.clf()
    plt.scatter( tsne_results[:,0], tsne_results[:,1], color=hue )
    plt.xlabel('tsne 2d comp1')
    plt.ylabel('tsne 2d comp2')
    plt.title(title)
    # details
    custom_lines = [Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], color='g', lw=2)
                    ]
    plt.legend(custom_lines, ['Prolate', 'Not-Prolate'], frameon=True)
    filename1 = 'tsne_2d_%s.png' % figlabel
    plt.savefig('%s/%s'%(fig_dir, filename1), format='png',
                bbox_inches='tight')
    plt.close('all')

    # plot 3d project
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*zip(*tsne_results), color=hue)
    plt.suptitle(title, fontsize=16, y=0.85)
    plt.legend(custom_lines, ['Prolate', 'Not-Prolate'], frameon=True)
    filename2 = 'tsne_3d_%s.png' % figlabel
    plt.savefig('%s/%s'%(fig_dir, filename2), format='png',
                bbox_inches='tight')
    plt.close('all')
    return [filename1, filename2]

# now save
fnames = plot_highd_data(features=feats.values,
                         targets=shape_data['shape'],
                         title='classification based on axis ratios',
                         figlabel='axis-ratios-based_shape%s' % Rdecider)
readme.update(to_write='Saved %s\n' % fnames)

# now work with shapes based on triaxiality and plot some things
shape_data = {}
T_20, T_40 = [], []
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
        shape_data['T_%s' % Rdecider] = [ data_now['T'][ind] ]
    else:
        shape_data['T_%s' % Rdecider] += [ data_now['T'][ind] ]
    #
    T_20 += [ data_now['T'][ np.where( data_now['Rstar'] == 20 )[0] ] ]
    T_40 += [ data_now['T'][ np.where( data_now['Rstar'] == 40 )[0] ] ]
    #
for key in shape_data:
    shape_data[key] = np.array(shape_data[key]).flatten()
#
T_20 = np.array(T_20).flatten()
T_40 = np.array(T_40).flatten()
# assemble classification based on triaxiality
threshold_T = 0.7
shape_class = {}
shape_class['shape'] = np.empty_like( shape_data['T_%s' % Rdecider] ).astype(str)
shape_class['shape'][:] = 'Not-P'
shape_class['shape'][ shape_data['T_%s' % Rdecider] > threshold_T] = 'P'
shape_data = shape_class
shape_data = pd.DataFrame( shape_data )
# now plot
fnames = plot_highd_data(features=feats.values,
                         targets=shape_data['shape'],
                         title='classification based on triaxiality',
                         figlabel='T-based_shape%s' % Rdecider)
readme.update(to_write='Saved %s\n' % fnames)
# ----------------------------------------------------------------------
# plot the analog of Hongyu's Fig 3.
for key in ['logm100', 'logm', 'logm30']:
    logmass = feats[key].values
    m_arr = np.arange( min(logmass) - 0.1, max(logmass) + 0.1, 0.1 )
    # set up
    fig, ax1 = plt.subplots()
    # plot normed densities
    ind1 = np.where( shape_data['shape'] == 'P' )[0]
    ax1.hist(logmass[ind1], histtype='step', bins=m_arr, color='r',
             lw=2, density=True )
    ind2 = np.where( shape_data['shape'] != 'P')[0]
    ax1.hist(logmass[ind2], histtype='step', bins=m_arr, color='b',
             lw=2, density=True )
    ax1.set_xlabel( key )
    ax1.set_ylabel( 'Normalized Number Densities' )
    # set up the second axis
    ax2 = ax1.twinx()
    # plot fraction
    frac, ms = [], []
    for j in range(len(m_arr) - 1):
        mlow, mupp = m_arr[j], m_arr[j+1]
        ind = np.where( ( logmass >= mlow ) & ( logmass < mupp ))[0]
        if len(ind) > 0:
            frac.append( len( set(ind1) & set(ind) ) / len( ind) )
            ms.append( np.median( [mlow, mupp ]) )
    ax2.plot(ms, frac, 'k.-', label='Prolate Fraction')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel( 'Prolate Fraction', rotation=-90, labelpad=20)
    ax2.grid(None)
    # details
    custom_lines = [Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], color='b', lw=2),
                    Line2D([0], [0], color='k', lw=2)]
    plt.legend(custom_lines,
               ['Prolate normed density (N=%s)' % len(ind1),
                'Not-Prolate normed density (N=%s)' % len(ind2),
                'Prolate Fraction'],
               bbox_to_anchor=(1.6,1), frameon=True)
    # save plot
    filename = 'hongyu_analog_%sgals_%s_shape%s_T-based.png' % (i+1, key, Rdecider)
    plt.savefig('%s/%s'%(fig_dir, filename), format='png',
                bbox_inches='tight')
    plt.close('all')
    readme.update(to_write='Saved %s\n' % filename)
# ----------------------------------------------------------------------
# plot ellipticity gradients vs triaxiality gradients
hue = np.array( ['r'] * len(T_20)  )
hue[ shape_class['shape'] == 'P' ] = 'g'
plt.clf()
plt.scatter( feats['e_37'] - feats['e_18'], T_40 - T_20, color=hue)
plt.xlabel( r'e$_{37}$ - e$_{18}$' )
plt.ylabel( r'T$_{40}$ - T$_{20}$' )
custom_lines = [Line2D([0], [0], color='g', lw=10),
                Line2D([0], [0], color='r', lw=10)]
plt.legend(custom_lines,
           [r'Prolate (T$_{%s}>$%s)' % ( Rdecider, threshold_T, ),
            r'Not-Prolate (T$_{%s}\leq$%s ' % ( Rdecider, threshold_T ) ],
           loc='best', frameon=True)
filename = 'egrad_vs_Tgrad_shape%s.png' % Rdecider
plt.savefig('%s/%s'%(fig_dir, filename), format='png',
            bbox_inches='tight')
plt.close('all')
readme.update(to_write='Saved %s\n' % filename)
# plot the actual values of ellipticity vs triaxiality
plt.clf()
plt.scatter( feats['e_37'], T_40, color=hue)
plt.xlabel( r'e$_{37}$' )
plt.ylabel( r'T$_{40}$' )
custom_lines = [Line2D([0], [0], color='g', lw=10),
                Line2D([0], [0], color='r', lw=10)]
plt.legend(custom_lines,
           [r'Prolate (T$_{%s}>$%s)' % ( Rdecider, threshold_T, ),
            r'Not-Prolate (T$_{%s}\leq$%s ' % ( Rdecider, threshold_T ) ],
           loc='best', frameon=True)
filename = 'e_vs_T_shape%s.png' % Rdecider
plt.savefig('%s/%s'%(fig_dir, filename), format='png',
            bbox_inches='tight')
plt.close('all')
readme.update(to_write='Saved %s\n' % filename)
# ----------------------------------------------------------------------
readme.update(to_write='## Time taken: %s\n'%get_time_passed(start_time))
