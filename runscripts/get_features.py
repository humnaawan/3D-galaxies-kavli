import pandas as pd
import numpy as np
import pickle
import datetime, time, socket, os
from d3g2d import run_rf, readme as readme_obj, get_time_passed
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--summary_datapath', dest='summary_datapath',
                  help='Path to the folder with the summary data.')
parser.add_option('--shape_datapath', dest='shape_datapath',
                  help='Path to the folder with the shape data.')
parser.add_option('--outdir', dest='outdir',
                  help='Path to the folder where to save results.')
parser.add_option('--q',
                  action='store_false', dest='quiet', default=False,
                  help='No print statements.')
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
summary_datapath = options.summary_datapath
shape_datapath = options.shape_datapath
outdir = options.outdir
quiet = options.quiet
# ------------------------------------------------------------------------------
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running get_features.py on %s\n\n' % socket.gethostname()
update += 'Options:\n%s\n' % options
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()

# start things up
start_time = time.time()
# ------------------------------------------------------------------------------
def get_features(data_for_halo):

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

    # store logm100
    ind = np.where( data_for_halo['aper_rkpc'] == 100 )[0]
    features['logm100'] = data_for_halo['aper_logms'][ind][0]

    ind = np.where( data_for_halo['aper_rkpc'] == 200 )[0]
    features['logm200'] = data_for_halo['aper_logms'][ind][0]

    return features
# ------------------------------------------------------------------------------
# read in shape data to get the haloIds
file = [ f for f in os.listdir(shape_datapath) if f.startswith('shape100_')][0]
with open('%s/%s' % (shape_datapath, file), 'rb') as f:
    shape = pickle.load(f)
# ------------------------------------------------------------------------------
# now read the summary data and assemble features
for i, haloId in enumerate(shape['haloId']):
    filename = [f for f in os.listdir(summary_datapath) if f.__contains__('_%s_'%haloId)]
    if len(filename) != 1:
        print(filename, haloId)
        break
    else:
        filename = filename[0]
    data = np.load('%s/%s' % (summary_datapath, filename))
    out = get_features(data_for_halo=data)
    if i == 0:
        keys = out.keys()
        feats = list(out.values())
    else:
        feats = np.vstack([feats, list(out.values())])
# assemble the features as a dataframe
feats = pd.DataFrame(feats, columns=keys)
# save the data
filename = 'features_%s.csv' % len( feats.keys() )
feats.to_csv('%s/%s' % (outdir, filename), index=False)
update = 'Saved %s in %s\n' % (filename, outdir)
update += '## Time taken: %s\n'%get_time_passed(start_time)
readme.update(to_write=update)
if not quiet: print(update)
