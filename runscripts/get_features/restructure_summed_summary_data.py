# this script flattens the 3-proj data into simpler data arrays, saved in
# separate folders for each projection
# https://github.com/dr-guangtou/riker/blob/master/demo/using_mass_map_summary.ipynb
import os
import numpy as np

illustris = False

if illustris:
    sim_name = 'illustris'
else:
    sim_name = 'tng'

summary_datapath = '/Users/humnaawan/repos/3D-galaxies-kavli/data/sum_%s/' % sim_name

dirs = {}
for proj in ['xy', 'yz', 'xz']:
    dirs[proj] = '%s/%s/' % (summary_datapath, proj)
    os.makedirs(dirs[proj], exist_ok=True)

for file in [ f for f in os.listdir(summary_datapath) if f.endswith('.npy') ]:
    haloId = file.split('_')[4]
    proj = file.split('_')[5]

    data = np.load('%s/%s' % (summary_datapath, file) )

    simpler = {
        # Subhalo ID
        'catsh_id': data['info']['catsh_id'],
        # Stellar mass of the galaxy
        'logms': data['info']['logms'],
        # Basic geometry of the galaxy
        'aper_x0': data['geom']['x'],
        'aper_y0': data['geom']['y'],
        'aper_ba': data['geom']['ba'],
        'aper_pa': data['geom']['pa'],
        # Aperture mass profile
        'aper_rkpc': data['aper']['rad_mid'],
        # Total mass enclosed in the aperture
        'aper_maper': data['aper']['maper_gal'],
        # Stellar mass in the bin
        'aper_mbins': data['aper']['mprof_gal'],
        # 1-D profile with varied shape
        'rkpc_shape': data['prof']['gal_shape']['r_kpc'],
        # This is the surface mass density profile and its error
        'mu_shape': data['prof']['gal_shape']['intens'] / (data['info']['pix'] ** 2.0),
        'mu_err_shape': data['prof']['gal_shape']['int_err'] / (data['info']['pix'] ** 2.0),
        # This is the ellipticity profiles
        'e_shape': data['prof']['gal_shape']['ell'],
        'e_err_shape': data['prof']['gal_shape']['ell_err'],
        # This is the normalized position angle profile
        'pa_shape': data['prof']['gal_shape']['pa_norm'],
        'pa_err_shape': data['prof']['gal_shape']['pa_err'],
        # Total mass enclosed by apertures
        'maper_shape': data['prof']['gal_shape']['growth_ori'],
        # 1-D profile using fixed shape
        'rkpc_prof': data['prof']['gal_mprof']['r_kpc'],
        # This is the surface mass density profile and its error
        'mu_mprof': data['prof']['gal_mprof']['intens'] / (data['info']['pix'] ** 2.0),
        'mu_err_mprof': data['prof']['gal_mprof']['int_err'] / (data['info']['pix'] ** 2.0),
        # Total mass enclosed by apertures
        'maper_mprof': data['prof']['gal_mprof']['growth_ori'],
        # Fourier deviations
        'a1_mprof': data['prof']['gal_mprof']['a1'],
        'a1_err_mprof': data['prof']['gal_mprof']['a1_err'],
        'a2_mprof': data['prof']['gal_mprof']['a2'],
        'a2_err_mprof': data['prof']['gal_mprof']['a2_err'],
        'a3_mprof': data['prof']['gal_mprof']['a3'],
        'a3_err_mprof': data['prof']['gal_mprof']['a3_err'],
        'a4_mprof': data['prof']['gal_mprof']['a4'],
        'a4_err_mprof': data['prof']['gal_mprof']['a4_err'],
        'b1_mprof': data['prof']['gal_mprof']['b1'],
        'b1_err_mprof': data['prof']['gal_mprof']['b1_err'],
        'b2_mprof': data['prof']['gal_mprof']['b2'],
        'b2_err_mprof': data['prof']['gal_mprof']['b2_err'],
        'b3_mprof': data['prof']['gal_mprof']['b3'],
        'b3_err_mprof': data['prof']['gal_mprof']['b3_err'],
        'b4_mprof': data['prof']['gal_mprof']['b4'],
        'b4_err_mprof': data['prof']['gal_mprof']['b4_err']
        }

    filename = '%s_summary_%s_%s.npy' % (sim_name, haloId, proj)
    np.save('%s/%s' % (dirs[proj], filename ), simpler)
