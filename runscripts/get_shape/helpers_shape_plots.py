import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np, pickle, os
from d3g2d import rcparams
for key in rcparams: mpl.rcParams[key] = rcparams[key]

def plot_shape_trends(outdir, data_dir, sim_name, shape_tag, Rdecider,
                      shape_class_data, colors_dict, classtag_to_classname, class_tag):
    if class_tag != '' and class_tag is not None: class_tag = '_%s' % class_tag
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
    alpha = 0.3
    # initiate counters
    counters = {}
    for key in colors_dict: counters[key] = 0
    # loop over the local foders; each folder is for a specific halo
    for i, folder in enumerate([f for f in os.listdir(data_dir) if f.startswith(sim_name)]):
        haloId = int(folder.split('halo')[-1].split('_')[0])
        ind = np.where(np.array( shape_class_data['haloId'] ) == haloId)[0]
        shape_class = np.array(shape_class_data['shape%s_class' % Rdecider])[ind][0]
        counters[shape_class] += 1
        # ---------------------------------------------------------------------
        # now read in the shape data
        file = [ f for f in os.listdir('%s/%s' % (data_dir, folder)) if f.startswith(shape_tag)]
        if len(file) != 1:
            raise ValueError('Somethings wrong: file array: %s' % file)
        # read in the file
        file = file[0]
        with open('%s/%s/%s' % (data_dir, folder, file), 'rb') as f:
            data = pickle.load(f)
        # calculate triaxiality
        data['T'] = (1 -  data['b/a'] ** 2 ) / (1 -  data['c/a'] ** 2 )
        # figure 1
        # plot b/a
        axes1[0].plot( data['Rstar'], data['b/a'], '.-', alpha=alpha, color=colors_dict[shape_class])
        # plot c/a
        axes1[1].plot( data['Rstar'], data['c/a'], '.-', alpha=alpha, color=colors_dict[shape_class])
        # plot triaxiality
        axes1[2].plot( data['Rstar'], data['T'], '.-',alpha=alpha, color=colors_dict[shape_class])
        # figure 2
        rmean = ( data['Rstar'][:-1] +  data['Rstar'][1:]) / 2
        # plot b/a
        axes2[0].plot( rmean, np.diff( data['b/a'] ), '.-', alpha=alpha, color=colors_dict[shape_class])
        # plot c/a
        axes2[1].plot( rmean, np.diff( data['c/a'] ), '.-', alpha=alpha, color=colors_dict[shape_class])
        # plot triaxiality
        axes2[2].plot( rmean, np.diff( data['T'] ), '.-', alpha=alpha, color=colors_dict[shape_class])
        # figure 3
        # plot b/a
        axes3[0].hist( data['b/a'], density=True,
                      histtype='step', lw=2, alpha=0.4, bins=bins, color=colors_dict[shape_class])
        # plot c/a
        axes3[1].hist( data['c/a'], density=True,
                      histtype='step', lw=2, alpha=0.4, bins=bins, color=colors_dict[shape_class])
        # plot triaxiality
        axes3[2].hist( data['T'],  density=True,
                      histtype='step', lw=2, alpha=0.4, bins=bins, color=colors_dict[shape_class])

    # finalize plots
    # figure 1
    axes1[0].set_title('Total: %s galaxies' % (i+1) )
    axes1[0].set_ylabel('b/a')
    axes1[1].set_ylabel('c/a')
    axes1[2].set_ylabel('T')
    axes1[2].set_xlabel('R (kpc)')
    fig1.set_size_inches(15, 15)
    # add legend
    custom_lines, class_labels = [], []
    for shape_class in colors_dict.keys():
        class_labels += ['%s (N=%s)' % (classtag_to_classname[shape_class], counters[shape_class]) ]
        custom_lines += [Line2D([0], [0], color=colors_dict[shape_class], lw=10) ]

    axes1[0].legend(custom_lines, class_labels,
                    bbox_to_anchor=(1, 0.9), frameon=False)
    # save figure
    filename = 'shape%s%s_trends.png' % (Rdecider, class_tag)
    fig1.savefig('%s/%s' % (outdir, filename), format='png',
                 bbox_inches='tight')
    plt.close(fig1)
    update = 'Saved %s\n' % filename
    # figure 2
    axes2[0].set_title('Total: %s galaxies' % (i+1) )
    axes2[0].set_ylabel(r'$\Delta$ b/a')
    axes2[1].set_ylabel(r'$\Delta$ c/a')
    axes2[2].set_ylabel(r'$\Delta$ T')
    axes2[2].set_xlabel('R (kpc)')
    fig2.set_size_inches(15, 15)
    # add legend
    axes2[0].legend(custom_lines, class_labels,
                    bbox_to_anchor=(1, 0.9), frameon=True)
    # save figure
    filename = 'shape%s%s_trends_differences.png' % (Rdecider, class_tag)
    fig2.savefig('%s/%s'%(outdir, filename), format='png',
                 bbox_inches='tight')
    plt.close(fig2)
    update += '\nSaved %s\n' % filename
    # figure 3
    axes3[0].set_xlabel('b/a')
    axes3[1].set_xlabel('c/a')
    axes3[2].set_xlabel('T')
    for i in range(3):
        axes3[i].set_ylabel('Counts')
    axes3[0].legend(custom_lines, class_labels,
                    bbox_to_anchor=(1, 0.9), frameon=False)
    fig3.set_size_inches(15, 15)
    filename = 'shape%s%s_histograms.png' % (Rdecider, class_tag)
    fig3.savefig('%s/%s'%(outdir, filename), format='png',
                 bbox_inches='tight')
    plt.close('all')
    update += '\nSaved %s\n' % filename
    return update
# ------------------------------------------------------------------------------
def plot_axis_ratios(outdir, shape_data, Rdecider,
                      shape_class_data, colors_dict, classtag_to_classname, class_tag):
    if class_tag != '' and class_tag is not None: class_tag = '_%s' % class_tag
    # also plot the axis ratios
    counters = {}
    plt.clf()
    for shape in colors_dict.keys():
        ind = np.where( shape_class_data['shape%s_class' % Rdecider] == shape )[0]
        counters[shape] = len(ind)
        plt.plot(shape_data['b/a_%s' % Rdecider][ind], shape_data['c/a_%s' % Rdecider][ind], '.',
                color=colors_dict[shape])
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
    custom_lines, class_labels = [], []
    for shape_class in colors_dict.keys():
        class_labels += ['%s (N=%s)' % (classtag_to_classname[shape_class], counters[shape_class]) ]
        custom_lines += [Line2D([0], [0], color=colors_dict[shape_class], lw=10) ]

    plt.legend(custom_lines, class_labels, loc='best', frameon=True)
    # save plot
    filename = 'axis_ratios_classification_shape%s%s.png' % (Rdecider, class_tag)
    plt.savefig('%s/%s'%(outdir, filename), format='png',
                bbox_inches='tight')
    plt.close('all')

    return 'Saved %s\n' % filename
