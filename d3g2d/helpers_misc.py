import time

__all__ = ['get_time_passed']

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
