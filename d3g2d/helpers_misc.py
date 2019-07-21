import time
import h5py

from .settings import h0

__all__ = ['get_time_passed', 'read_cutout']

# ------------------------------------------------------------------------------
def get_time_passed(time0, string=True):
    """

    This function calculates the time passed since input time0

    Required Input
    ---------------
    * time0: float: starting time.

    Optional Input
    ---------------
    * string: bool: True if want <time passed> <units>; else return both.

    Returns
    -------
    if string=True:
        * time_passed: float: time passed since time0
        * units: str: units of time_passed
    else:
        * str: <time passed> <units>

    """
    time_passed= time.time()-time0  # in s
    units = 's'

    if (time_passed > 60.):  # > 1 min
        time_passed /= 60.  # now in min
        units = 'min'
        if (time_passed > 60.): # > 1hr
            time_passed /= 60.  # now in min
            units = 'hrs'
    if string:
        return '%.2f %s'%(time_passed, units)
    else:
        return time_passed, units

def read_cutout(fname, z, PartType=4, key='Coordinates'):
    """
    keys:
      dm - Coordinates, Velocities
      stars - Coordinates, Velocities, GFM_Metallicity,
              GFM_StellarFormationTime, GFM_StellarPhotometrics, Masses,
    units are convert to physical units (i.e. no h0 and scale factor a)

    From https://github.com/HongyuLi2016/illustris-tools/blob/435dceb93802547394b1257228c724b2a502d4cd/utils/util_illustris.py#L159
    """
    scale_factor = 1.0 / (1+z)
    with h5py.File(fname, 'r') as f:
        Ptype = 'PartType{}'.format(PartType)
        data = f[Ptype][key][:]
    if key == 'Coordinates':
        # distance in kpc
        data *= scale_factor/h0
    elif key == 'Velocities':
        # velocity in km/s
        data *= np.sqrt(scale_factor)
    elif key == 'Masses':
        # mass in M_sun
        data /= h0
    return data
