import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import confusion_matrix

from .settings import rcparams
for key in rcparams: mpl.rcParams[key] = rcparams[key]

__all__ = ['plot_confusion_matrix']

# ------------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, classes, normalize, save_plot,
                          outdir=None, fig_label=None, readme=None):
    """
    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if save_plot and outdir is None:
        raise ValueError('Must specify outdir to save plot.')
    # set up the title
    if fig_label is None: fig_label = ''
    if fig_label.startswith('_'): title_label = fig_label[1:]
    else: title_label = fig_label
    if normalize:
        title = 'Normalized confusion matrix\n%s' % title_label
    else:
        title = '(Unnormalized) Confusion matrix\n%s' % title_label

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        update = 'Normalized confusion matrix'
    else:
        update = 'Unnormalized confusion matrix'
    update += '\n%s' % cm
    if readme is not None:
        readme.update(to_write=update)
    # --------------------------------------------------------------------------
    # now plot
    cmap = plt.cm.Blues
    plt.clf()
    fig, ax = plt.subplots()
    if normalize:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes, title=title,
           ylabel='True label', xlabel='Predicted label')
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_plot:
        if normalize: tag = 'normed'
        else: tag = 'unnormed'
        # set up filename
        filename = 'plot_confusion_matrix_%s%s.png' % (tag, fig_label)
        # save file
        plt.savefig('%s/%s'%(outdir, filename), format='png',
                    bbox_inches='tight')
        plt.close('all')
        update = 'Saved %s\n' % filename
        if readme is not None:
            readme.update(to_write=update)
    else:
        plt.show()

    return
