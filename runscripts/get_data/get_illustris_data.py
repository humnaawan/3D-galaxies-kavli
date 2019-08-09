import datetime, time, socket, os
import numpy as np
from d3g2d import get_data, get_time_passed, readme as readme_obj

# ------------------------------------------------------------------------------
illustris_z0 = False  # True for z=0.0 data; False for z=0.4 one
# ------------------------------------------------------------------------------
if illustris_z0:
    path = '/Users/humnaawan/repos/3D-galaxies-kavli/data/illustris_mass_shape/mass-all-11p0/'
    z = 0.0
    snap_num = 135
    # get the haloIds
    haloIds = []
    for file in os.listdir(path):
        haloIds.append( int( file.split('subhalo')[1].split('.dat')[0] ) )
    haloIds = np.unique( haloIds )
else:
    # get z=0.4 cutouts
    path = '/Users/humnaawan/repos/3D-galaxies-kavli/data/sum_illustris/'
    z = 0.4
    snap_num = 108
    # get the haloIds
    haloIds = []
    for file in os.listdir(path):
        haloIds.append( int( file.split('_')[4] ) )
    haloIds = np.unique( haloIds )

# set up
run_name = 'Illustris-1'
outdir = '/Users/humnaawan/repos/3D-galaxies-kavli/outputs/illustris_z%s/' % z
# ------------------------------------------------------------------------------
# set up the readme
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running on %s\n\n' % socket.gethostname()
update += 'Outdir: %s\n' % outdir
update += 'For z = %s, run_name = %s\n' % (z, run_name)
update += '%s haloIds:\n%s\n' % ( len(haloIds), haloIds)
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()

# save the halo ids
filename = 'haloIds.txt'
np.savetxt('%s/%s' % (outdir, filename), haloIds, fmt='%s')
readme.update(to_write='Saved %s' % filename)

# now get the cutouts etc
get_data(run_name=run_name, z=z, snap_num=snap_num, haloIds=haloIds,
         outdir=outdir, print_progress=True, readme=readme)

readme.update(to_write='Done.\n## Time taken: %s\n' % get_time_passed(start_time) )
