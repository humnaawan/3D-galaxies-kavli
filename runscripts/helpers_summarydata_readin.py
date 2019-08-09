import numpy as np

# ------------------------------------------------------------------------------
def get_features_highres_summed(data_for_halo):
    data_for_halo = data_for_halo[()]
    features = {}
    for key in ['aper_ba']:
        features[key] = data_for_halo[key]

    for pair in [[9, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17]]:
        ind_inner, ind_outer = pair
        inner_r = data_for_halo['aper_rkpc'][ind_inner]
        outer_r = data_for_halo['aper_rkpc'][ind_outer]
        M_in = data_for_halo['aper_maper'][ind_inner]
        M_out = data_for_halo['aper_maper'][ind_outer]
        features['gradM_%.f_%.f' % (inner_r, outer_r)] = (M_out - M_in) / M_out

    for pair in [[16, 18], [18, 20], [20, 21], [21, 22], \
                 [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29] ]:
        ind_inner, ind_outer = pair
        inner, outer = data_for_halo['rkpc_shape'][ind_inner], data_for_halo['rkpc_shape'][ind_outer]

        e_in =  data_for_halo['e_shape'][ind_inner]
        e_out = data_for_halo['e_shape'][ind_outer]

        features['dele_%.f_%.f' % (inner, outer)] = e_out - e_in

        pa_in =  data_for_halo['pa_shape'][ind_inner]
        pa_out = data_for_halo['pa_shape'][ind_outer]
        del_pa = pa_out - pa_in
        if del_pa > 45: del_pa -= 45
        if del_pa < -45: del_pa += 45

        features['delpa_%.f_%.f' % (inner, outer)] = del_pa

    for ind in range(16, 30):
        features['a1_%.f' % (data_for_halo['rkpc_shape'][ind])] = data_for_halo['a1_mprof'][ind]
        features['a4_%.f' % (data_for_halo['rkpc_shape'][ind])] = data_for_halo['a4_mprof'][ind]
    # add ellipticity close to 100kpc
    for ind in range(16,29):
        rval = data_for_halo['rkpc_shape'][ind]
        features['e_%.f' % rval] = data_for_halo['e_shape'][ind]
    # add logm100
    features['logm100'] = np.log10( data_for_halo['aper_maper'][-2] )

    features['logm'] = data_for_halo['logms']
    features['logm30'] = np.log10( data_for_halo['aper_maper'][-6] )

    return features
# ------------------------------------------------------------------------------
def get_features_highres(data_for_halo):

    features = {}
    for key in ['aper_ba']:
        features[key] = data_for_halo[key]

    for pair in [[7, 9], [9, 11], [11, 13], [13, 14], [14, 15]]:
        ind_inner, ind_outer = pair
        inner_r = data_for_halo['aper_rkpc'][ind_inner]
        outer_r = data_for_halo['aper_rkpc'][ind_outer]
        M_in = 10 ** data_for_halo['aper_logms'][ind_inner]
        M_out = 10 ** data_for_halo['aper_logms'][ind_outer]
        features['gradM_%.f_%.f' % (inner_r, outer_r)] = (M_out - M_in) / M_out

    for pair in [[17, 21], [21, 24], [24, 28], [28, 30]]:
        ind_inner, ind_outer = pair
        inner, outer = data_for_halo['rkpc_shape'][ind_inner], data_for_halo['rkpc_shape'][ind_outer]

        e_in =  data_for_halo['e_shape'][ind_inner]
        e_out = data_for_halo['e_shape'][ind_outer]

        features['dele_%.f_%.f' % (inner, outer)] = e_out - e_in

        pa_in =  data_for_halo['pa_shape'][ind_inner]
        pa_out = data_for_halo['pa_shape'][ind_outer]
        del_pa = pa_out - pa_in
        if del_pa > 45: del_pa -= 45
        if del_pa < -45: del_pa += 45

        features['delpa_%.f_%.f' % (inner, outer)] = del_pa

    for ind in [17, 21, 24, 28, 29]:
        features['a1_%.f' % (data_for_halo['rkpc_shape'][ind])] = data_for_halo['a1_shape'][ind]
        features['a4_%.f' % (data_for_halo['rkpc_shape'][ind])] = data_for_halo['a4_shape'][ind]
    # add ellipticity close to 100kpc
    for ind in range(17,31):
        rval = data_for_halo['rkpc_shape'][ind]
        features['e_%.f' % rval] = data_for_halo['e_shape'][ind]
    # add logm100
    features['logm100'] = data_for_halo['aper_logms'][-2]

    features['logm'] = data_for_halo['logms']
    features['logm30'] = data_for_halo['aper_logms'][-6]

    return features
# ------------------------------------------------------------------------------
def get_features_lowres(data_for_halo):

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
    # add ellipticity close to 100kpc
    ind = 20
    rval = data_for_halo['rpix_shape'][ind] * 5.333
    features['e_%.f' % rval] = data_for_halo['e_shape'][ind]
    # add logm100
    ind = np.where( data_for_halo['aper_rkpc'] == 100 )[0]
    features['logm100'] = data_for_halo['aper_logms'][ind][0]
    # add logm200
    ind = np.where( data_for_halo['aper_rkpc'] == 200 )[0]
    features['logm200'] = data_for_halo['aper_logms'][ind][0]

    return features
