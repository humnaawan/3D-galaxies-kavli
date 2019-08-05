import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import pickle, datetime, time, socket, os

from d3g2d import run_rf, readme as readme_obj, get_time_passed, rcparams
for key in rcparams: mpl.rcParams[key] = rcparams[key]
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
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
summary_datapath = options.summary_datapath
shape_datapath = options.shape_datapath
outdir = options.outdir
low_res = options.low_res
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
def get_features_highres(data_for_halo):

    features = {}
    for key in ['aper_ba']:
        features[key] = data_for_halo[key]

    for pair in [[7, 9], [9, 11], [11, 13], [13, 14], [14, 15]]:
        ind_inner, ind_outer = pair
        inner_r = data_for_halo['aper_rkpc'][ind_inner]
        outer_r = data_for_halo['aper_rkpc'][ind_outer]
        M_in = 10 ** data_for_halo['aper_logms'][ind_inner]
        M_out = 10 ** data_for_halo['aper_logms'][ind_outer]
        features['delM_%.f_%.f' % (inner_r, outer_r)] = (M_out - M_in) / M_out

    for pair in [[17, 21], [21, 24], [24, 28], [28, 30]]:
        ind_inner, ind_outer = pair
        inner, outer = data_for_halo['rkpc_shape'][ind_inner], data_for_halo['rkpc_shape'][ind_outer]

        e_in =  data_for_halo['e_shape'][ind_inner]
        e_out = data_for_halo['e_shape'][ind_outer]

        features['dele_%.f_%.f' % (inner, outer)] = e_out - e_in

        pa_in =  data_for_halo['pa_shape'][ind_inner]
        pa_out = data_for_halo['pa_shape'][ind_outer]
        del_pa = pa_out - pa_in
        if del_pa > 45: del_pa -= 45
        if del_pa < -45: del_pa += 45

        features['delpa_%.f_%.f' % (inner, outer)] = del_pa

    for ind in [17, 21, 24, 28, 29]:
        features['a1_%.f' % (data_for_halo['rkpc_shape'][ind])] = data_for_halo['a1_shape'][ind]
        features['a4_%.f' % (data_for_halo['rkpc_shape'][ind])] = data_for_halo['a4_shape'][ind]
    # add ellipticity close to 100kpc
    for ind in range(17,31):
        rval = data_for_halo['rkpc_shape'][ind]
        features['e_%.f' % rval] = data_for_halo['e_shape'][ind]
    # add logm100
    features['logm100'] = data_for_halo['aper_logms'][-2]

    features['logm'] = data_for_halo['logms']
    features['logm30'] = data_for_halo['aper_logms'][-6]

    return features
# ------------------------------------------------------------------------------
def get_features_lowres(data_for_halo):

    features = {}
    for key in ['aper_ba']:
        features[key] = data_for_halo[key]

    for pair in [[10, 50], [50, 100], [100, 150], [150, 200]]:
        inner, outer = pair
        ind_inner = np.where(data_for_halo['aper_rkpc'] == inner)[0]
        ind_outer = np.where(data_for_halo['aper_rkpc'] == outer)[0]
        M_in = 10 ** data_for_halo['aper_logms'][ind_inner][0]
        M_out = 10 ** data_for_halo['aper_logms'][ind_outer][0]
        features['delM_%s_%s' % (inner, outer)] = (M_out - M_in) / M_out

    for pair in [[8, 12], [12, 16], [16, 20],
                 [20, 24], [24, 28], [28, 30]]:
        ind_inner, ind_outer = pair
        inner, outer = data_for_halo['rpix_shape'][ind_inner] * 5.333, data_for_halo['rpix_shape'][ind_outer] * 5.333

        e_in =  data_for_halo['e_shape'][ind_inner]
        e_out = data_for_halo['e_shape'][ind_outer]

        features['dele_%.f_%.f' % (inner, outer)] = e_out - e_in

        pa_in =  data_for_halo['pa_shape'][ind_inner]
        pa_out = data_for_halo['pa_shape'][ind_outer]
        del_pa = pa_out - pa_in
        if del_pa > 45: del_pa -= 45
        if del_pa < -45: del_pa += 45

        features['delpa_%.f_%.f' % (inner, outer)] = del_pa
    # add ellipticity close to 100kpc
    ind = 20
    rval = data_for_halo['rpix_shape'][ind] * 5.333
    features['e_%.f' % rval] = data_for_halo['e_shape'][ind]
    # add logm100
    ind = np.where( data_for_halo['aper_rkpc'] == 100 )[0]
    features['logm100'] = data_for_halo['aper_logms'][ind][0]
    # add logm200
    ind = np.where( data_for_halo['aper_rkpc'] == 200 )[0]
    features['logm200'] = data_for_halo['aper_logms'][ind][0]

    return features
# ------------------------------------------------------------------------------
# read in shape data to get the haloIds
file = [ f for f in os.listdir(shape_datapath) if f.startswith('shape100_')][0]
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
filename = 'mass_profiles_%sgals.png' % (i+1)
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
filename = 'mass_profiles_%sgals_wm100quartiles.png' % (i+1)
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
    filename = 'hongyu_analog_%sgals_%s.png' % (i+1, key)
    plt.savefig('%s/%s'%(fig_dir, filename), format='png',
                bbox_inches='tight')
    plt.close('all')
    readme.update(to_write='Saved %s\n' % filename)

# ----------------------------------------------------------------------
readme.update(to_write='## Time taken: %s\n'%get_time_passed(start_time))
