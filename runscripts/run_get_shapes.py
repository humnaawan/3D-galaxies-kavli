import datetime, time, socket
import h5py
from d3g2d import get_shape_main, readme as readme_obj, get_time_passed
from d3g2d import tng_snap2z, illustris_snap2z, felipes_datapath
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
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
data_dir = options.data_dir
quiet = options.quiet
test = options.test
# ------------------------------------------------------------------------------
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running on %s\n\n' % socket.gethostname()
update += 'data_dir: %s\n' % data_dir
update += '\nOptions:\n%s\n' % options
readme = readme_obj(outdir=data_dir, readme_tag=readme_tag, first_update=update)
readme.run()

if test:
    halo_ids = [5, 16941]
    z = illustris_snap2z['z']
    sim_name = 'Illustris-1'
else:
    with h5py.File(felipes_datapath, 'r') as f:
        halo_ids = f['catsh_id'][:]
    z = tng_snap2z['z']
    sim_name = 'TNG100-1'

for halo_id in halo_ids:
    start_time = time.time()
    update = 'Getting shape data for halo %s'  % halo_id
    readme.update(to_write=update)
    if not quiet: print(update)
    filename = get_shape_main(source_dir='%s/%s_halo%s_z%s' % (data_dir, sim_name, halo_id, z),
                              fname='cutout_%s.hdf5' % halo_id,
                              test_illustris=test)
    update = 'Saved %s\n' % filename
    update += '## Time taken: %s\n'%get_time_passed(start_time)
    readme.update(to_write=update)
    if not quiet: print(update)
