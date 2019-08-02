import datetime, time, socket, os
import numpy as np
from d3g2d import get_shape_main, readme as readme_obj, get_time_passed
from d3g2d import tng_snap2z, illustris_snap2z, summary_datapath, get_shape_class
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
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
data_dir = options.data_dir
quiet = options.quiet
test = options.test
illustris = options.illustris
# ------------------------------------------------------------------------------
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running on %s\n\n' % socket.gethostname()
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

# now classify
update = 'Getting shape classification ... \n'
start_time = time.time()
filename = get_shape_class(data_dir=data_dir, startwith_tag=sim_name,
                           shape_tag=shape_tag, Rdecider=100)
update += 'Saved %s\n' % filename
update += '## Time taken: %s\n'%get_time_passed(start_time)
readme.update(to_write=update)
if not quiet: print(update)
