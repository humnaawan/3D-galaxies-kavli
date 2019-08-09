import datetime, time, socket, os
import numpy as np
from d3g2d import get_data, get_time_passed, readme as readme_obj

# ------------------------------------------------------------------------------
outdir = '/Users/humnaawan/repos/3D-galaxies-kavli/outputs/tng-100_z0.4/'
run_name = 'TNG100-1'
z = 0.4
snap_num = 72

# get the haloIds.
summary_datapath = '/Users/humnaawan/repos/3D-galaxies-kavli/data/tng_lowres_2d_summary/'
haloIds = []
for f in [f for f in os.listdir(summary_datapath) if f.endswith('.npy')]:
    haloIds.append( int(f.split('_')[1]) )
# ------------------------------------------------------------------------------
# set up the readme
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running on %s\n\n' % socket.gethostname()
update += 'Outdir: %s\n' % outdir
update += 'For z = %s, run_name = %s\n' % (z, run_name)
update += '%s haloIds:\n%s\n' % ( len(haloIds), haloIds )
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()

# save the halo ids
filename = 'haloIds.txt'
np.savetxt('%s/%s' % (outdir, filename), haloIds, fmt='%s')
readme.update(to_write='Saved %s' % filename)

# get the data.
get_data(run_name=run_name, z=z, snap_num=snap_num, haloIds=haloIds,
         outdir=outdir, print_progress=True, readme=readme)

readme.update(to_write='Done.\n## Time taken: %s\n' % get_time_passed(start_time) )
