import pickle
import numpy as np
import numpy.linalg as LA
import os
import pandas as pd

from .helpers_misc import read_cutout
from .settings import h0

__all__ = ['get_shape_main', 'get_shape_class']

def shape(R, xpart):
    """
    R can be a vector of radius to measure shape, and xpart is just the coordinate array.

    From https://github.com/HongyuLi2016/illustris-tools/blob/435dceb93802547394b1257228c724b2a502d4cd/shape/illustris-get_shape.py#L10-L22
    """
    mpart = np.ones(xpart.shape[0])
    axisRatios = np.zeros([len(R), 2])
    eigenVectors = np.zeros([len(R), 3, 3])
    for i in range(len(R)):
        ba, ca, angle, Tiv = get_shape(xpart, mpart, Rb=R[i])
        for j in range(3):
            if Tiv[j, 2] < 0:
                Tiv[j, :] *= -1
        # eigen vectors Tiv[0, :] Tiv[1, :] Tiv[2, :]
        axisRatios[i, :] = np.array([ba, ca])
        eigenVectors[i, :, :] = Tiv
    return axisRatios, eigenVectors

#
def get_shape(x, mpart, Rb=20., decrease=True):
    # https://github.com/HongyuLi2016/illustris-tools/blob/435dceb93802547394b1257228c724b2a502d4cd/utils/util_illustris.py#L398-L481
    # Rb=20kpc, within which the shape is calcualted
    s = 1.
    q = 1.

    Tiv = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    # tmp = Tiv
    order = [2, 1, 0]
    Vei = np.zeros((3, 3))
    dq = 10000.
    ds = 10000.

    while (dq > 0.01 or ds > 0.01):
        # in eigenvector coordinates
        y = np.transpose(np.dot(Tiv, np.transpose(x)))
        rn0 = np.sqrt(np.power(y[:, order[2]], 2.) +
                      np.power(y[:, order[1]], 2.)/q/q +
                      np.power(y[:, order[0]], 2.)/s/s)
        ind = np.where(rn0 < Rb)[0]
        # Np = ind.shape[0]

        y1 = y[ind, 0]
        y2 = y[ind, 1]
        y3 = y[ind, 2]
        rn = rn0[ind]

        I11 = np.sum(y1*y1/np.power(rn, 2))
        I22 = np.sum(y2*y2/np.power(rn, 2))
        I33 = np.sum(y3*y3/np.power(rn, 2))
        I12 = np.sum(y1*y2/np.power(rn, 2))
        I13 = np.sum(y1*y3/np.power(rn, 2))
        I23 = np.sum(y2*y3/np.power(rn, 2))

        II = [[I11, I12, I13],
              [I12, I22, I23],
              [I13, I23, I33]]

        D, A = LA.eig(II)
        # print 'eigenvalues'
        # print D
        # print  'eigenvectors'
        # print A
        order = np.argsort(D)  # a=order2,b=order1,c=order0
        la = np.sqrt(D[order[2]])
        lb = np.sqrt(D[order[1]])
        lc = np.sqrt(D[order[0]])

        dq = np.abs(q-lb/la)
        ds = np.abs(s-lc/la)

        q = lb/la
        s = lc/la

        Tiv = np.dot(LA.inv(A), Tiv)

    # rba = q
    # rca = s
    if decrease:
        Tiv = Tiv[order[::-1], :]
    else:
        Tiv = Tiv[order, :]
    # right hand coordinate
    if np.dot(Tiv[2, :], np.cross(Tiv[0, :], Tiv[1, :])) < 0:
        Tiv[1, :] *= -1
    # eigen vectors Vei[:,0] Vei[:,1] Vei[:,2]
    Vei = LA.inv(Tiv)

    d = np.array([0, 0, 1])
    costh = np.dot(Vei[:, 2], d) / np.sqrt(np.dot(Vei[:, 2], Vei[:, 2])) /\
        np.sqrt(np.dot(d, d))
    # angle between longest axis (z' direction) and LOS (i.e. z direction)
    angle = np.arccos(costh)*180./np.pi

    ba = q
    ca = s
    # print ' b/a= {:.2f}'.format(q),'  c/a = {:.2f}'.format(s)
    # print "rotation angle=",angle

    return ba, ca, angle, Tiv

def get_shape_main(source_dir, z, fname, illustris=False, Rstar=None):
    """
    Modified from:
    https://github.com/HongyuLi2016/illustris-tools/blob/435dceb93802547394b1257228c724b2a502d4cd/shape/illustris-get_shape.py#L25-L49
    info pickle is not nested anymore
    care only about the star part
    specify Rstar by hand

    """
    # set some things up
    if illustris:
        if Rstar is None:
            Rstar = np.arange(1, 101, 1)
    else:
        if Rstar is None:
            Rstar = np.arange(20, 160, 10)
    # get some data
    with open('%s/info.dat' % source_dir, 'rb') as f:
        info = pickle.load(f)
    pos = np.array(info['SubhaloPos']) / (1 + z) / h0
    # read in the coorinates
    xpart_star = read_cutout(fname='%s/%s' % (source_dir, fname), z=z) - pos
    # calculate stuff
    axis_ratios_star, evectors_star = shape(Rstar, xpart_star)
    # set up and save data
    rst = {'Rstar': Rstar,
           'b/a': axis_ratios_star[:, 0],
           'c/a': axis_ratios_star[:, 1],
           }
    filename = 'shape_%sRvals.dat' % (len(Rstar))
    with open('%s/%s' % (source_dir, filename), 'wb') as f:
        pickle.dump(rst, f)

    return filename

def get_shape_class(outdir, shape_data_dict, classification_type, Rdecider, threshold_T=0.7):
    allowed = [ 'axis-ratios-based', 'T-based', 'fe-based' ]
    if classification_type not in allowed:
        raise ValueError( 'classification_type must be one of the following: %s; input: %s' % (allowed, classification_type) )
    # if axis_ratios_based = True: classification based on c/a, b/a
    # otherwise based on triaxiality
    # shape_data_dict should be a pandas dataframe
    shape = np.array( ['undecided'] * len( shape_data_dict ) , dtype=str)
    if classification_type == 'axis-ratios-based':
        arr_ba = shape_data_dict['b/a_%s' % Rdecider].values
        arr_ca = shape_data_dict['c/a_%s' % Rdecider].values

        shape[:] = 'S'
        # prolates
        ind = np.where( (( arr_ba - arr_ca) < 0.2 ) & (arr_ba < 0.8 ) )[0]
        shape[ind] = 'P'
        # triaxials
        ind = np.where( (( arr_ba - arr_ca) > 0.2 ) & (arr_ba < 0.8 ) )[0]
        shape[ind] = 'T'
        # oblates
        ind = np.where( (( arr_ba - arr_ca) > 0.2 ) & (arr_ba > 0.8 ) )[0]
        shape[ind] = 'O'
    elif classification_type == 'T-based':
        shape[:] = 'Not-P'
        arr_T = shape_data_dict['T_%s' % Rdecider].values
        ind_prolate = np.where(  arr_T > threshold_T )[0]
        shape[ind_prolate] = 'P'
    elif classification_type == 'fe-based':
        arr_f = shape_data_dict['flattening_%s' % Rdecider].values
        arr_e = shape_data_dict['elongation_%s' % Rdecider].values
        shape[:] = 'S'
        # oblates
        ind = np.where( (arr_f >= 0.5 ) & (arr_e <= 0.5 ) )[0]
        shape[ind] = 'O'
        # triaxials
        ind = np.where( (arr_f >= 0.5 ) & (arr_e >= 0.5 ) )[0]
        shape[ind] = 'T'
        # prolates
        ind = np.where( (arr_f < 0.5 ) & (arr_e > 0.5 ) )[0]
        shape[ind] = 'P'
    else:
        raise ValueError( 'Somethings wrong. Input classification_type: %s' % classification_type )
    # assemble a dataframe
    data = pd.DataFrame( { 'shape%s_class' % Rdecider: shape,
                          'haloId' : shape_data_dict['haloId'] } )
    tag = classification_type
    if classification_type == 'T-based':
        tag = 'T-based_%sthres' % threshold_T
    filename = 'shape%s_classes_%s_%shaloIds.csv' % (Rdecider, tag, len(shape_data_dict['haloId']) )

    data.to_csv('%s/%s' % (outdir, filename), index=False)

    return filename
