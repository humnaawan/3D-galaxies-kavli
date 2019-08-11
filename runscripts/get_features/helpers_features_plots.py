import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from d3g2d import rcparams
for key in rcparams: mpl.rcParams[key] = rcparams[key]

__all__ = ['summary_data_plots', 'plot_hongyus_analog', 'plot_highd_data']

def summary_data_plots(outdir, summary_datapath, Rdecider,
                      shape_class_data, colors_dict, classtag_to_classname,
                      class_tag, summed_data, mass_cut=False, mass_thresh=11.4):
    if class_tag != '' and class_tag is not None: class_tag = '_%s' % class_tag
    # read in one data set to figure things out
    haloId = shape_class_data['haloId'][0]
    filename = [f for f in os.listdir(summary_datapath) if f.__contains__('_%s_'%haloId)]
    if len(filename) != 1:
        raise ValueError('Somethings wrong: file array: %s' % file)
    filename = filename[0]
    data = np.load('%s/%s' % (summary_datapath, filename))
    if summed_data:
        data = data[()]
    # things to plot vs rpix_shape
    toplot_pixshape = [f for f in data.keys() if f.endswith('_shape') \
                       and not f.__contains__('err') \
                       and not f.__contains__('output') \
                       and not f.startswith('a') \
                       and not f.startswith('b') \
                       and not f.__contains__('rpix_shape') \
                       and not f.__contains__('rkpc_shape')
                       ]
    print('\nPlotting rpix_shape vs. %s' % toplot_pixshape )

    toplot_pixsprof = [f for f in data.keys() if f.endswith('prof') \
                       and not f.__contains__('err') \
                       and not f.__contains__('output') \
                       and not f.startswith('a') \
                       and not f.startswith('b') \
                       and not f.__contains__('rpix_prof') \
                       and not f.__contains__('rkpc_prof')
                       ]

    print('\nPlotting rpix_prof vs. %s' % toplot_pixsprof)

    toplot_pixsprof_2nd_order = [f for f in data.keys() if f.endswith('prof') \
                                and ( f.startswith('a') or f.startswith('b') ) \
                                and not f.__contains__('err') \
                                and not f.__contains__('rpix_prof') \
                                and not f.__contains__('rkpc_prof')
                                ]

    print('\nPlotting rpix_prof vs. %s' % toplot_pixsprof_2nd_order)

    # figure for the rpix_shape plot
    nrows, ncols = len(toplot_pixshape), 1
    fig1, axes1 = plt.subplots(nrows, ncols)
    fig1.set_size_inches(15, 5 * nrows)
    plt.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
    # figure for the rpix_prof plot; no 2nd order moments
    nrows, ncols = len(toplot_pixsprof), 1
    fig2, axes2 = plt.subplots(nrows, ncols)
    fig2.set_size_inches(15, 5 * nrows)
    plt.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
    # figure aper_rkpc', 'aper_logms
    fig3, axes3 = plt.subplots(1, 1)
    fig3.set_size_inches(10, 5)
    plt.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)
    # plot 2nd order moments, if any
    plot_2nd_order = False
    if len(toplot_pixsprof_2nd_order) > 0:
        plot_2nd_order = True
        nrows, ncols = len(toplot_pixsprof_2nd_order), 1
        fig4, axes4 = plt.subplots(nrows, ncols)
        fig4.set_size_inches(15, 5 * nrows)
        plt.subplots_adjust(wspace=0.3, hspace=0.2, top=0.9)

    alpha = 0.3
    shape_class_data['shape%s_class' % Rdecider] = np.array(shape_class_data['shape%s_class' % Rdecider])

    # also plot the axis ratios
    counters = {}
    for key in colors_dict: counters[key] = 0
    gal_count = 0
    # loop over the halos
    for i, haloId in enumerate(shape_class_data['haloId']):
        filename = [f for f in os.listdir(summary_datapath) if f.__contains__('_%s_'%haloId)]
        if len(filename) != 1:
            raise ValueError('Somethings wrong: file array: %s' % file)
        filename = filename[0]
        data = np.load('%s/%s' % (summary_datapath, filename))
        if summed_data:
            data = data[()]
            mass_key = 'aper_maper'
            data['aper_maper'] = np.log10( data['aper_maper'] )
        else:
            mass_key = 'aper_logms'

        # check if need to implement a mass cut
        if mass_cut:
            # check ~ logm30
            rthres = data['aper_rkpc'][-6]
            del_mass = 0.1
            if ( ( data[mass_key][-6] >= mass_thresh - del_mass ) and \
                           ( data[mass_key][-6] <= mass_thresh + del_mass  ) ):
                plot = True
            else:
                plot = False
        else:
            plot = True
        # plot
        if plot:
            gal_count += 1
            # see if need to implement a mass cut
            # find the shape
            shape_class = shape_class_data['shape%s_class' % Rdecider][i]
            # add to the right counter
            counters[shape_class] += 1
            # plot _shape related things
            for j, key in enumerate( toplot_pixshape ):
                axes1[j].plot( data['rkpc_shape'] ** 0.25, data[key], '.-',
                              alpha=alpha, color=colors_dict[shape_class])
            # plot _prof related things
            for j, key in enumerate( toplot_pixsprof ):
                axes2[j].plot( data['rkpc_prof'] ** 0.25, data[key], '.-',
                              alpha=alpha, color=colors_dict[shape_class])
            if plot_2nd_order:
                for j, key in enumerate( toplot_pixsprof_2nd_order ):
                    axes4[j].plot( data['rkpc_prof'], data[key], '.-',
                                  alpha=alpha, color=colors_dict[shape_class])
            # plot logm
            axes3.plot( data['aper_rkpc'], data[mass_key], '.-',
                       alpha=alpha, color=colors_dict[shape_class])
    # plot details
    tag = ''  # for the filename
    title = ''
    if mass_cut:
        title += r'%.2f $\leq$ logM%.2f $\leq$ %.2f; $\Delta$M: %s' % (mass_thresh - del_mass,
                                                                       rthres,
                                                                       mass_thresh + del_mass,
                                                                       del_mass)
        title += '\n\n'
        tag = '_mass-cut-%s' % mass_thresh
    percent_tag = ''
    if mass_cut:
        percent_tag = ' (%.2f%% of total)' % ( gal_count/(i+1) * 100)
    title += 'Total: %s galaxies%s' % (gal_count, percent_tag)
    # set title
    axes1[0].set_title( title )
    axes2[0].set_title( title )
    axes3.set_title( title )
    # axis labels
    for j, key in enumerate( toplot_pixshape ):
        axes1[j].set_ylabel( key )
        if key.__contains__('mu_') or key.__contains__('maper'):
            axes1[j].set_yscale('log')
    axes1[-1].set_xlabel( r'( rkpc_shape )$^{1/4}$' )
    #
    for j, key in enumerate( toplot_pixsprof ):
        axes2[j].set_ylabel( key )
        if key.__contains__('mu_') or key.__contains__('maper'):
            axes2[j].set_yscale('log')
    axes2[-1].set_xlabel( r' ( rkpc_prof )$^{1/4}$' )

    axes3.set_ylabel(mass_key)
    axes3.set_xlabel('aper_rkpc')

    # add legend
    custom_lines, class_labels = [], []
    for shape_class in colors_dict.keys():
        class_labels += ['%s (N=%s)' % (classtag_to_classname[shape_class], counters[shape_class]) ]
        custom_lines += [ Line2D([0], [0], color=colors_dict[shape_class], lw=10) ]
    axes1[0].legend(custom_lines, class_labels, loc='best', frameon=True)
    axes2[0].legend(custom_lines, class_labels, loc='best', frameon=True)
    axes3.legend(custom_lines, class_labels, loc='best', frameon=True)
    # save figure
    filename = 'rshape_profiles_%sgals_shape%s%s%s.png' % (gal_count, Rdecider, class_tag, tag)
    fig1.savefig('%s/%s' % (outdir, filename), format='png',
                 bbox_inches='tight')
    plt.close(fig1)
    update = 'Saved %s\n' % filename

    # save figure
    filename = 'rprof_profiles_%sgals_shape%s%s%s.png' % (gal_count, Rdecider, class_tag, tag)
    fig2.savefig('%s/%s' % (outdir, filename), format='png',
                 bbox_inches='tight')
    plt.close(fig2)
    update += 'Saved %s\n' % filename

    # save figure
    filename = 'aper_profiles_%sgals_shape%s%s%s.png' % (gal_count, Rdecider, class_tag, tag)
    fig3.savefig('%s/%s' % (outdir, filename), format='png',
                 bbox_inches='tight')
    plt.close(fig3)
    update += 'Saved %s\n' % filename

    if plot_2nd_order:
        axes4[0].set_title( title )
        for j, key in enumerate( toplot_pixsprof_2nd_order ):
            axes4[j].set_ylabel( key )
        axes4[-1].set_xlabel( 'rkpc_prof' )
        axes4[0].legend(custom_lines, class_labels, loc='best', frameon=True)
        # save plot
        filename = 'rprof_profiles_2ndorder_%sgals_shape%s%s%s.png' % (gal_count, Rdecider, class_tag, tag)
        fig4.savefig('%s/%s' % (outdir, filename), format='png',
                     bbox_inches='tight')
        plt.close(fig4)
        update += 'Saved %s\n' % filename

    return update

# ------------------------------------------------------------------------------
def plot_hongyus_analog(outdir, logmass, shape_class_arr, Rdecider, class_tag, mass_tag):
    if class_tag != '' and class_tag is not None: class_tag = '_%s' % class_tag
    m_arr = np.arange( min(logmass) - 0.1, max(logmass) + 0.1, 0.1 )
    # set up
    fig, ax1 = plt.subplots()
    # plot normed densities
    ind1 = np.where( shape_class_arr == 'P' )[0]
    ax1.hist(logmass[ind1], histtype='step', bins=m_arr, color='r',
             lw=2, density=False )
    ind2 = np.where( shape_class_arr != 'P')[0]
    ax1.hist(logmass[ind2], histtype='step', bins=m_arr, color='b',
             lw=2, density=False )
    ax1.set_xlabel( mass_tag )
    ax1.set_ylabel( 'Galaxy Counts' )
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
               ['Prolate Counts (N=%s)' % len(ind1),
                'Not-Prolate Counts (N=%s)' % len(ind2),
                'Prolate Fraction'],
               bbox_to_anchor=(1.5,1), frameon=True)
    # save plot
    filename = 'hongyu_analog_%sgals_shape%s%s_%s.png' % (len(logmass), Rdecider, class_tag, mass_tag)
    plt.savefig('%s/%s'%(outdir, filename), format='png',
                bbox_inches='tight')
    plt.close('all')
    return 'Saved %s\n' % filename

# ----------------------------------------------------------------------
# plot tSNE plots
def plot_highd_data(outdir, features, targets, title, figlabel):
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
    plt.savefig('%s/%s'%(outdir, filename1), format='png',
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
    plt.savefig('%s/%s'%(outdir, filename2), format='png',
                bbox_inches='tight')
    plt.close('all')

    return [filename1, filename2]
