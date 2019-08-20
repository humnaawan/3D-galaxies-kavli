import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import operator
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import os
os.environ["PYTHONWARNINGS"] = "ignore:Liblinear failed to converge:UserWarning:sklearn.svm.base"

from .helpers_plot import plot_confusion_matrix
from .settings import rcparams
for key in rcparams: mpl.rcParams[key] = rcparams[key]

__all__ = ['run_rf', 'inverse_onehot', 'get_chisq', 'get_mse', 'evaluate_model']

def run_rf(feats, feat_labels, targets, target_labels, outdir,
           regression=False, readme=None, n_feats_to_return=None):
    # ----------------------------------------------------------------------
    if not regression:
        target_labels = np.unique( targets )
        # employ one-hot encoding
        # ----------------------------------------------------------------------
        update = 'Running one-hot encoding; have %s targets classes' % len( np.unique( targets) )
        if readme is not None:
            readme.update(to_write=update)
        # ----------------------------------------------------------------------
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(targets)
        targets = pd.DataFrame(encoder.transform(targets).toarray(), columns=encoder.categories_ ).values
    else:
        encoder = None
    # --------------------------------------------------------------------------
    # split data
    x_train, x_test, y_train, y_test = train_test_split(feats, targets,
                                                        test_size=0.33,
                                                        random_state=42)
    arr = np.array( y_train )
    all_tot = len( arr )
    update = '## %s objects in train set\n' % all_tot
    if encoder is None:
        targets_ = arr
    else:
        targets_ = encoder.inverse_transform( arr )

    for target in np.unique( targets_ ):
        tot_target = len( np.where( targets_ == target)[0] )
        update += '%s of %s (%.2f%% of total)\n' % (tot_target, target, 100 * tot_target/all_tot)

    arr = np.array( y_test )
    all_tot = len(arr)
    update += '\n## %s objects in test set\n' % all_tot
    if encoder is None:
        targets_ = arr
    else:
        targets_ = encoder.inverse_transform(arr)

    print(np.unique( targets_ ))
    for target in np.unique( targets_ ):
        tot_target = len( np.where( targets_ == target)[0] )
        update += '%s of %s (%.2f%% of total)\n' % (tot_target, target, 100 * tot_target/all_tot)

    if readme is not None:
        readme.update(to_write=update)
    # adapting https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    # to find the best params.
    # ----------------------------------------------------------------------------
    # first try with no optimization
    update = '\n# ----------------------------------------------------------------------------'
    update += '\n## Trying RF with no optimization ... \n'
    rf = RandomForestRegressor(n_estimators=20, verbose=0, random_state=42)
    rf.fit(x_train, y_train)
    update += 'param values: %s\n' % rf.get_params()
    ## cross-validation
    kfold = KFold(n_splits=10, random_state=42)
    scores = cross_val_score(rf, x_train, y_train, cv=kfold)
    update += '\n# cross val score: accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
    if readme is not None:
        readme.update(to_write=update)
    # fit the model to training set first
    _ = evaluate_model(model=rf, x_arr=x_train, y_arr=y_train, save_plot=True,
                       data_label='train_set_rf_unoptimized', outdir=outdir,
                       encoder=encoder, feat_labels=feat_labels,
                       target_labels=target_labels,
                       regression=regression, readme=readme)
    # now fit the model to test set
    _ = evaluate_model(model=rf, x_arr=x_test, y_arr=y_test, save_plot=True,
                       data_label='test_set_rf_unoptimized', outdir=outdir,
                       encoder=encoder, feat_labels=feat_labels,
                       target_labels=target_labels,
                       regression=regression, readme=readme)
    # ----------------------------------------------------------------------------
    # now try optimization
    update = '\n# ----------------------------------------------------------------------------\n'
    update += '## Trying coarse-grained optimization ... \n'
    if readme is not None:
        readme.update(to_write=update)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=500, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap
                   }
    update = 'param grid: \n%s\n' % random_grid
    if readme is not None:
        readme.update(to_write=update)

    # run over the param grid
    rf = RandomForestRegressor(random_state=42)
    # random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=5, verbose=-2, random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(x_train, y_train)
    update = 'Best params: %s\n' % rf_random.best_params_
    update += '\nCV results:\n'
    for key in [f for f in rf_random.cv_results_ if f.__contains__('score') and not f.__contains__('time')]:
        update += '%s: %s\n' % ( key, list( rf_random.cv_results_[key] ) )

    ## cross-validation
    kfold = KFold(n_splits=10, random_state=42)
    scores = cross_val_score(rf_random.best_estimator_, x_train, y_train, cv=kfold)
    update += '\n# cross val score: accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
    if readme is not None:
        readme.update(to_write=update)

    # lets make a plot of the cv results
    plt.clf()
    for key in [f for f in rf_random.cv_results_ if f.__contains__('score') and \
                not f.__contains__('time') and \
                f.__contains__('split')]:
        plt.plot( rf_random.cv_results_[key], '.', label=key )
    niter = len(rf_random.cv_results_[key])
    plt.errorbar(range( niter ),
                 rf_random.cv_results_['mean_test_score'], fmt='o', color='k',
                 yerr=rf_random.cv_results_['std_test_score'], label='mean + std')
    plt.xlabel( 'niter' )
    plt.ylabel( 'CV scores' )
    plt.legend(bbox_to_anchor=(1,1))
    ax1 = plt.gca()
    ax1.set_xlim(-1, niter+1)

    ax1.set_xticks( range(niter) )
    ax1.set_xticklabels( range(niter) )
    ax1.tick_params(axis='x', labelsize=8)
    ax2 = ax1.twiny()
    ax2.set_xlim(-1, niter+1)
    ax2.set_xticks( ax1.get_xticks() )
    ax2.set_xticklabels( rf_random.cv_results_['rank_test_score'] )
    ax2.tick_params(axis='x', labelsize=8)
    ax2.grid( None )
    ax2.set_xlabel('rank test score')
    plt.gcf().set_size_inches(25, 8)
    filename = 'plot_cv_scores.png'
    # save file
    plt.savefig('%s/%s'%(outdir, filename), format='png',
                bbox_inches='tight')
    plt.close('all')
    update = 'Saved %s' % filename
    if readme is not None:
        readme.update(to_write=update)


    # evaluate model
    model = rf_random.best_estimator_
    # fit the model to training set first
    _ = evaluate_model(model=model, x_arr=x_train, y_arr=y_train, save_plot=True,
                       data_label='train_set_rf_optimized', outdir=outdir,
                       encoder=encoder, feat_labels=feat_labels,
                       target_labels=target_labels,
                       regression=regression, readme=readme)
    # now fit the model to test set
    y_pred  = evaluate_model(model=model, x_arr=x_test, y_arr=y_test, save_plot=True,
                             data_label='test_set_rf_optimized', outdir=outdir,
                             encoder=encoder, feat_labels=feat_labels,
                             target_labels=target_labels,
                             regression=regression, readme=readme)
    # save the model
    if regression:
        filename = 'rf_regression_model_optimized.pickle'
    else:
        filename = 'rf_classifier_model_optimized.pickle'
    with open('%s/%s' % (outdir, filename), 'wb') as f:
        pickle.dump({'model': model}, f)
    # update
    update = 'Saved %s' % filename
    if readme is not None:
        readme.update(to_write=update)

    if n_feats_to_return is not None and (n_feats_to_return > 0) :
        n_feats_to_return = int( n_feats_to_return )
        if readme is not None:
            readme.update(to_write='Returning top %s features.\n' % n_feats_to_return )
        # access top <n_feats_to_return> features to return.
        f = dict(zip(feat_labels, np.array(model.feature_importances_)))
        ordered_feats = sorted( f.items(), key=operator.itemgetter(1) )
        # separate out the feature labels
        feats = []
        update = ''
        for i in ordered_feats[-n_feats_to_return:]:
            feats += [ i[0] ]
            update += '%s: %.2f\n' % (i[0], i[1])

        if readme is not None:
            readme.update(to_write=update )
        return np.array(feats)
    else:
        return

# ------------------------------------------------------------------------------
def inverse_onehot(encoder, y_test, y_pred):
    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)
    return y_test, y_pred
# ------------------------------------------------------------------------------
def get_chisq(true_arr, model_arr):
    true_arr, model_arr = np.array(true_arr), np.array(model_arr)
    # need to have len(true)==len(model)
    if len(true_arr) != len(model_arr):
        raise ValueError('true_arr and model_arr must be of the same length.')
    # calculate the statistic and return it
    return np.sum( ( true_arr - model_arr ) **2 / true_arr ) / len(true_arr)
# ------------------------------------------------------------------------------
def get_mse(test_arr, pred_arr):
    if len(np.shape(test_arr)) == 1:
        test_arr = np.reshape(test_arr, ( len(test_arr), 1) )
    if len(np.shape(pred_arr)) == 1:
        pred_arr = np.reshape(pred_arr, ( len(pred_arr), 1) )
    target_dim = np.size(test_arr, axis=1)

    if (target_dim > 1):
        mse = np.zeros(target_dim)
        for i in range(target_dim):
            mse[i] = mean_squared_error(test_arr[:, i], pred_arr[:, i])
    else:
        mse = [mean_squared_error(test_arr, pred_arr)]

    return mse
# ------------------------------------------------------------------------------
def evaluate_model(model, x_arr, y_arr, save_plot, feat_labels, target_labels,
                   data_label=None, outdir=None, encoder=None,
                   regression=False, readme=None):
    if save_plot and outdir is None:
        raise ValueError('Must specify outdir is want to save plots.')

    if data_label is None: data_label = ''
    # predict
    y_pred = model.predict(x_arr)
    # see how well the model is working.
    update = '\n#####################################\n%s\n' % data_label
    update += '#####################################\n'
    update +=  'RF performance:\n'
    if regression:
        update += 'R2 score = %s\n' % model.score(x_arr, y_arr)   # best possible score is 1.0
        update += 'MSE: %s\n' % get_mse(y_arr, y_pred)
    if feat_labels is None:
        update += '## Feature importance:\n%s\n' % np.array(model.feature_importances_)
    else:
        f = dict(zip(feat_labels, np.array(model.feature_importances_)))
        update += '## Feature importance:\n %s\n' % sorted(f.items(), key=operator.itemgetter(1))
    update += '#####################################\n'
    if readme is not None:
        readme.update(to_write=update)
    # plot more test; need to invert one hot encoding
    if regression:
        if len( np.shape(y_pred) ) == 1:
            y_pred = np.reshape( y_pred, (len(y_pred) , 1)  )
        if len( np.shape(y_arr) ) == 1:
            y_arr = np.reshape( y_arr, (len(y_arr), 1)  )
        ntargets = np.shape(y_pred)[-1]

        label = ''
        plt.clf()
        nrows, ncols = ntargets, 1
        fig, axes = plt.subplots(nrows, ncols)
        plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9)

        for i in range(ntargets):
            if ntargets == 1: ax = axes
            else: ax = axes[i]
            ax.plot(y_arr[:, i], y_pred[:, i], 'o')
            ax.plot(y_arr[:, i], y_arr[:, i], '-')
            ax.set_title(r'%s ; $\chi^2$/dof = %.2e' % ( target_labels[i],
                                                        get_chisq(true_arr=y_arr[:, i], model_arr=y_pred[:, i]) ) )
            if i == 0:
                label = target_labels[i].replace('/', '').replace('_', '')
            else:
                label += '_%s' % target_labels[i].replace('/', '').replace('_', '')
        fig.set_size_inches(ntargets * 5, ntargets * 5)

        if ntargets == 1:
            plt.suptitle(data_label, fontweight="bold", y=1.05)
            axes.set_xlabel('true' )
            axes.set_ylabel('predicted')
        else:
            plt.suptitle(data_label, fontweight="bold", y=0.95)
            axes[-1].set_xlabel('true' )
            for row in range( nrows ):
                axes[row].set_ylabel('predicted')

        if save_plot:
            if data_label != '': data_label = '_%s' % data_label
            # set up filename
            filename = 'plot_scatter_%s%s.png' % (label, data_label)
            # save file
            plt.savefig('%s/%s'%(outdir, filename), format='png',
                        bbox_inches='tight')
            plt.close('all')
            update = 'Saved %s' % filename
            if readme is not None:
                readme.update(to_write=update)
        else:
            plt.show()
    else:
        if encoder is not None:
            y_arr_, y_pred_ = inverse_onehot(encoder=encoder, y_test=y_arr, y_pred=y_pred)
        else:
            y_arr_, y_pred_ = y_arr, y_pred
        # set up to plot confusion matrix
        y_arr_ = list( np.reshape(y_arr_.flatten(), (len(y_arr_), )) )
        y_pred_ = list( np.reshape(y_pred_.flatten(), (len(y_pred_), )) )

        for normalize in [True, False]:
            plot_confusion_matrix(y_true=y_arr_, y_pred=y_pred_,
                                  classes=target_labels,
                                  normalize=normalize, save_plot=save_plot,
                                  outdir=outdir, fig_label=data_label,
                                  readme=readme)
        # print report
        update = classification_report( y_arr_, y_pred_)
        update +=  '\nAccuracy: %.f%% \n'%( accuracy_score(y_arr_, y_pred_) * 100 )

        if readme is not None:
            readme.update(to_write=update)
    return y_pred
