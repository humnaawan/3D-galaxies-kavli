import datetime, time, socket
from d3g2d import get_shape_main, readme as readme_obj, get_time_passed
# ------------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--data_dir', dest='data_dir',
                  help='Path to the folder with the data.')
parser.add_option('--q',
                  action='store_false', dest='quiet', default=False,
                  help='No print statements.')
# ------------------------------------------------------------------------------
(options, args) = parser.parse_args()
data_dir = options.data_dir
quiet = options.quiet
# ------------------------------------------------------------------------------
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running on %s\n\n' % socket.gethostname()
update += 'data_dir: %s\n' % data_dir
readme = readme_obj(outdir=data_dir, readme_tag=readme_tag, first_update=update)
readme.run()

halo_ids = [5, 16941]
for halo_id in halo_ids:
    start_time = time.time()
    update = 'Getting shape data for halo %s'  % halo_id
    readme.update(to_write=update)
    if not quiet: print(update)
    filename = get_shape_main(source_dir='%s/Illustris-1_halo%s_z0' % (data_dir, halo_id),
                              fname='cutout_%s.hdf5' % halo_id)
    update = 'Saved %s\n' % filename
    update += '## Time taken: %s\n'%get_time_passed(start_time)
    readme.update(to_write=update)
    if not quiet: print(update)
