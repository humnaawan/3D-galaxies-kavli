import datetime, time, socket
from d3g2d import get_data, readme as readme_obj

# ------------------------------------------------------------------------------
outdir = '/Users/humnaawan/repos/3D-galaxies-kavli/outputs/test_illustris/'
run_name = 'Illustris-1'
z = 0
haloIDs = [16941, 5]
# ------------------------------------------------------------------------------
# set up the readme
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running on %s\n\n' % socket.gethostname()
update += 'Outdir: %s\n' % outdir
update += 'For z = %s, run_name = %s\n' % (z, run_name)
update += 'haloIDs: %s\n' % haloIDs
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()

get_data(run_name=run_name, z=z, haloIDs=haloIDs,
         outdir=outdir, print_progress=True, readme=readme)
