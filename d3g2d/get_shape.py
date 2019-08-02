import pickle
import numpy as np
import numpy.linalg as LA
import os

from .helpers_misc import read_cutout
from .settings import h0, illustris_snap2z, tng_snap2z

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

def get_shape_main(source_dir, fname, illustris=False, Rstar=None):
    """
    Modified from:
    https://github.com/HongyuLi2016/illustris-tools/blob/435dceb93802547394b1257228c724b2a502d4cd/shape/illustris-get_shape.py#L25-L49
    info pickle is not nested anymore
    care only about the star part
    specify Rstar by hand

    """
    # set some things up
    if illustris:
        z = illustris_snap2z['z'] # replaces snapznum output in Hongyu's code
        if Rstar is None:
            Rstar = np.arange(1, 101, 1)
    else:
        z = tng_snap2z['z']
        if Rstar is None:
            Rstar = np.arange(20, 160, 10)
    # get some data
    with open('%s/info.dat'%source_dir, 'rb') as f:
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

def get_shape_class(data_dir, shape_tag, startwith_tag='TNG', Rdecider=100):
    shapes, haloIds = [], []
    # loop over the local foders; each folder is for a specific halo
    for i, folder in enumerate([f for f in os.listdir(data_dir) if f.startswith(startwith_tag)]):
        # ---------------------------------------------------------------------
        # now read in the data produced from my version
        file = [ f for f in os.listdir('%s/%s' % (data_dir, folder)) if f.startswith(shape_tag)]
        haloId =  int(folder.split('halo')[-1].split('_')[0])
        if len(file) == 1:
            file = file[0]
            with open('%s/%s/%s' % (data_dir, folder, file), 'rb') as f:
                data_now = pickle.load(f)

            ind = np.where( data_now['Rstar'] == Rdecider )[0]
            if ( ( data_now['b/a'][ind] - data_now['c/a'][ind]) < 0.2 ) and (data_now['b/a'][ind] < 0.8):
                shapes.append('P')
            elif ( ( data_now['b/a'][ind] - data_now['c/a'][ind]) > 0.2 ) and (data_now['b/a'][ind] < 0.8):
                shapes.append('T')
            elif ( ( data_now['b/a'][ind] - data_now['c/a'][ind]) > 0.2 ) and (data_now['b/a'][ind] > 0.8):
                shapes.append('O')
            else:
                shapes.append('S')
            # append the id too
            haloIds.append( haloId )
        else:
            raise ValueError('Somethings wrong: %s for halo %s' % ( file, haloId ) )

    filename = 'shape%s_classes_%shaloIds.pickle' % (Rdecider, len(haloIds) )
    rst = {'haloId': np.array( haloIds ),
           'shape': np.array( shapes )
           }
    with open('%s/%s' % (data_dir, filename), 'wb') as f:
        pickle.dump(rst, f)
    return filename
