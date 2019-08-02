import datetime, time, socket, os
from d3g2d import get_data, readme as readme_obj, illustris_snap2z

# ------------------------------------------------------------------------------
summary_datapath = '/Users/humnaawan/repos/3D-galaxies-kavli/data/illustris_mass_shape/mass-all-11p0/'
outdir = '/Users/humnaawan/repos/3D-galaxies-kavli/outputs/test_illustris/'
run_name = 'Illustris-1'
z = illustris_snap2z['z']
#haloIDs = [16941, 5]
haloIDs = []
for f in [f for f in os.listdir(summary_datapath) if f.endswith( '.dat' )]:
    haloIDs.append( int( f.split('subhalo')[1].split('.dat')[0] ) )
# ------------------------------------------------------------------------------
# set up the readme
start_time = time.time()
readme_tag = ''
update = '%s\n' % datetime.datetime.now()
update += 'Running on %s\n\n' % socket.gethostname()
update += 'Outdir: %s\n' % outdir
update += 'For z = %s, run_name = %s\n' % (z, run_name)
update += '%s haloIDs:\n%s\n' % ( len(haloIDs), haloIDs)
readme = readme_obj(outdir=outdir, readme_tag=readme_tag, first_update=update)
readme.run()

get_data(run_name=run_name, z=z, haloIDs=haloIDs,
         outdir=outdir, print_progress=True, readme=readme)
