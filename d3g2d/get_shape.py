import pickle
import numpy as np
import numpy.linalg as LA
import h5py

from .settings import h0

__all__ = ['get_shape_main']

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

def snap2z(snapNum):
    """
    from https://github.com/HongyuLi2016/illustris-tools/blob/435dceb93802547394b1257228c724b2a502d4cd/utils/util_illustris.py#L30-L86
    """
    snapNum = np.atleast_1d(snapNum)
    z = [4.67730473e+01, 4.45622038e+01, 4.24536738e+01, 4.06395569e+01,
         3.87125594e+01, 3.68747395e+01, 3.51219704e+01, 3.36139397e+01,
         3.20120740e+01, 3.04843396e+01, 2.90273057e+01, 2.76377005e+01,
         2.64421253e+01, 2.51721572e+01, 2.39609608e+01, 2.28058162e+01,
         2.18119639e+01, 2.07562707e+01, 1.97494329e+01, 1.87891896e+01,
         1.79630246e+01, 1.70854528e+01, 1.62484933e+01, 1.54502666e+01,
         1.47634960e+01, 1.40339921e+01, 1.33382483e+01, 1.26747021e+01,
         1.20418635e+01, 1.14973880e+01, 1.09190332e+01, 1.03674436e+01,
         9.99659047e+00, 9.84138044e+00, 9.38877127e+00, 9.00233985e+00,
         8.90799919e+00, 8.44947629e+00, 8.01217295e+00, 7.59510715e+00,
         7.23627607e+00, 7.00541705e+00, 6.85511726e+00, 6.49159775e+00,
         6.14490120e+00, 6.01075740e+00, 5.84661375e+00, 5.52976581e+00,
         5.22758097e+00, 4.99593347e+00, 4.93938066e+00, 4.66451770e+00,
         4.42803374e+00, 4.00794511e+00, 3.70877426e+00, 3.49086137e+00,
         3.28303306e+00, 3.08482264e+00, 3.00813107e+00, 2.89578501e+00,
         2.73314262e+00, 2.57729027e+00, 2.44422570e+00, 2.31611074e+00,
         2.20792547e+00, 2.10326965e+00, 2.00202814e+00, 1.90408954e+00,
         1.82268925e+00, 1.74357057e+00, 1.66666956e+00, 1.60423452e+00,
         1.53123903e+00, 1.47197485e+00, 1.41409822e+00, 1.35757667e+00,
         1.30237846e+00, 1.24847261e+00, 1.20625808e+00, 1.15460271e+00,
         1.11415056e+00, 1.07445789e+00, 1.03551045e+00, 9.97294226e-01,
         9.87852811e-01, 9.50531352e-01, 9.23000816e-01, 8.86896938e-01,
         8.51470901e-01, 8.16709979e-01, 7.91068249e-01, 7.57441373e-01,
         7.32636182e-01, 7.00106354e-01, 6.76110411e-01, 6.44641841e-01,
         6.21428745e-01, 5.98543288e-01, 5.75980845e-01, 5.46392183e-01,
         5.24565820e-01, 5.03047523e-01, 4.81832943e-01, 4.60917794e-01,
         4.40297849e-01, 4.19968942e-01, 3.99926965e-01, 3.80167867e-01,
         3.60687657e-01, 3.47853842e-01, 3.28829724e-01, 3.10074120e-01,
         2.91583240e-01, 2.73353347e-01, 2.61343256e-01, 2.43540182e-01,
         2.25988386e-01, 2.14425036e-01, 1.97284182e-01, 1.80385262e-01,
         1.69252033e-01, 1.52748769e-01, 1.41876204e-01, 1.25759332e-01,
         1.09869940e-01, 9.94018026e-02, 8.38844308e-02, 7.36613847e-02,
         5.85073228e-02, 4.85236300e-02, 3.37243719e-02, 2.39744284e-02,
         9.52166697e-03, 2.22044605e-16]
    number = [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
              10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
              20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
              30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
              40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
              50,  51,  52,  54,  56,  57,  58,  59,  60,  61,
              62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
              72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
              82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
              92,  93,  94,  95,  96,  97,  98,  99,  100, 101,
              102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
              112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
              122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
              132, 133, 134, 135]
    z = np.asarray(z)
    number = np.asarray(number)
    ii = number == snapNum
    if ii.sum() == 1:
        return z[ii][0]
    else:
        return np.nan

def get_shape_main(source_dir, fname):
    """
    Modified from:
    https://github.com/HongyuLi2016/illustris-tools/blob/435dceb93802547394b1257228c724b2a502d4cd/shape/illustris-get_shape.py#L25-L49
    info pickle is not nested anymore
    care only about the star part
    specify Rstar by hand

    """
    with open('%s/info.dat'%source_dir, 'rb') as f:
        info = pickle.load(f)
    # get some data
    snap_num = int(info['SnapshotNumber'])
    z = snap2z(snap_num)
    pos = np.array(info['SubhaloPos']) / (1 + z) / h0

    xpart_star = read_cutout('%s/%s' % (source_dir, fname), PartType=4,
                             key='Coordinates', z=z) - pos

    Rstar = np.arange(1, 101, 1)
    axis_ratios_star, evectors_star = shape(Rstar, xpart_star)

    rst = {'Rstar': Rstar,
           'b/a': axis_ratios_star[:, 0],
           'c/a': axis_ratios_star[:, 1],
           }

    filename = 'shape_%sRvals.dat' % (len(Rstar))
    with open('%s/%s' % (source_dir, filename), 'wb') as f:
        pickle.dump(rst, f)

    return filename
