################################################################################
# oops/meshgrid.py: Class Meshgrid
################################################################################

import numpy as np
import numbers

from polymath import Qube, Boolean, Scalar, Pair, Vector
from polymath import Vector3, Matrix3, Quaternion

from oops.fov import FOV

class Meshgrid(object):
    """A Meshgrid is an arbitrary array of coordinate pairs within a Field of
    View. It caches information about the line of sight and various derivatives,
    preventing the need for repeated calls to the FOV functions when the same
    field of view describes multiple images.

    After you create a Meshgrid object, the following are available as
    properties:

    uv              the (u,v) pairs with no derivatives.
    uv_w_derivs     the (u,v) pairs with d_dlos.
    duv_dlos        the partial derivatives d(u,v)/dlos.

    los             the line-of-sight unit vectors with no derivatives.
    los_w_derivs    the line-of-sight unit vectors with d_duv.
    dlos_duv        the partial derivatives dlos/d(u,v).
    """

    #===========================================================================
    def __init__(self, fov, uv_pair, fov_keywords={}):
        """The Meshgrid constructor.

        Input:
            fov         a FOV object.
            uv_pair     a Pair object of arbitrary shape, representing (u,v)
                        coordinates within a field of view.
            fov_keywords  an optional dictionary of parameters passed to the
                        FOV methods, containing parameters that might affect
                        the properties of the FOV.
        """

        self.fov = fov
        self.uv = Pair.as_pair(uv_pair).wod
        self.uv_w_duv_duv = self.uv.with_deriv('uv', Pair.IDENTITY, 'insert')
        self.fov_keywords = fov_keywords
        self.shape = self.uv.shape

        # For holding info if time is a single value or None
        self.filled_los_w_derivs = {}
        self.filled_los = {}
        self.filled_uv_w_derivs = {}

    def __getstate__(self):
        return (self.fov, self.uv, self.fov_keywords)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def for_fov(fov, origin=0.5, undersample=1, oversample=1, limit=None,
                     swap=False, fov_keywords={}):
        """A 2-D rectangular Meshgrid object for a specified sampling of the
        FOV.

        Input:
            origin      A single value, tuple or Pair defining the origin of the
                        grid. Default is 0.5, which places the first sample in
                        the middle of the first pixel.

            limit       A single value, tuple or Pair defining the upper limits
                        of the meshgrid. By default, this is the shape of the
                        FOV.

            undersample A single value, tuple or Pair defining the magnitude of
                        under-sampling to be performed. For example, a value of
                        2 would cause the meshgrid to sample every other pixel
                        along each axis.

            oversample  A single value, tuple or Pair defining the magnitude of
                        over-sampling to be performed. For example, a value of
                        2 would create a 2x2 array of samples inside each pixel.

            swap        True to swap the order of the indices in the meshgrid,
                        (v,u) instead of (u,v).

            fov_keywords  an optional dictionary of parameters passed to the
                        FOV methods, containing parameters that might affect
                        the properties of the FOV.
        """

        # Convert inputs to NumPy 2-element arrays
        if limit is None:
            limit = fov.uv_shape
        if isinstance(limit, numbers.Real):
            limit = (limit,limit)
        limit = Pair.as_pair(limit).values.astype('float')

        if isinstance(origin, numbers.Real):
            origin = (origin, origin)
        origin = Pair.as_pair(origin).values.astype('float')

        if isinstance(undersample, numbers.Real):
            undersample = (undersample, undersample)
        undersample = Pair.as_pair(undersample).values.astype('float')

        if isinstance(oversample, numbers.Real):
            oversample = (oversample, oversample)
        oversample = Pair.as_pair(oversample).values.astype('float')

        # Construct the 1-D index arrays
        step = undersample/oversample
        limit = limit + step * 1.e-10   # Allow a little slop at the upper end

        urange = np.arange(origin[0], limit[0], step[0])
        vrange = np.arange(origin[1], limit[1], step[1])

        # Construct the 2-D index arrays
        usize = urange.size
        vsize = vrange.size

        if usize == 1:
            urange = np.array(urange[0])
        if vsize == 1:
            vrange = np.array(vrange[0])

        grid = Pair.combos(urange, vrange).values

        # Swap axes if necessary
        if usize > 1 and vsize > 1 and swap:
            grid = grid.swapaxes(0,1)

        return Meshgrid(fov, grid, fov_keywords)

    #===========================================================================
    @staticmethod
    def _as_key(time):
        """Given time as a key to the internal dictionaries. False if not
        suitable as a key.
        """

        if time is None:
            return time

        if isinstance(time, numbers.Real):
            return time

        if isinstance(time, Scalar) and np.isscalar(time.vals):
            return time.vals

        return False

    #===========================================================================
    def los_w_derivs(self, time=None):

        # Return from internal dictionary if present
        try:
            return self.filled_los_w_derivs[time]
        except (KeyError, TypeError):
            pass

        # Evaluate the LOS anew
        los_ = self.fov.los_from_uvt(self.uv_w_duv_duv, time=time, derivs=True,
                                     **self.fov_keywords)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self.filled_los_w_derivs[key] = los_

        return los_

    #===========================================================================
    def los(self, time=None):

        # Return from internal dictionary if present
        # If it's in filled_los_w_derivs, adapt that one
        try:
            return self.filled_los[time]
        except KeyError:        # on a KeyError, we can try the other dict
            try:
                los_wod = self.filled_los_w_derivs[time].wod
            except KeyError:    # not found, so give up with dicts
                pass
            else:               # found, so strip the derivs, save and return
                self.filled_los[time] = los_wod
                return los_wod
        except TypeError:
            pass

        # Evaluate the LOS anew
        los_ = self.fov.los_from_uvt(self.uv, time=time, derivs=False,
                                     **self.fov_keywords)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self.filled_los[key] = los_

        return los_

    #===========================================================================
    def dlos_duv(self, time=None):
        return self.los_w_derivs(time).d_duv

    #===========================================================================
    def uv_w_derivs(self, time=None):

        # Return from internal dictionary if present
        try:
            return self.filled_uv_w_derivs[time]
        except (KeyError, TypeError):
            pass

        # Evaluate (u,v) anew
        los_ = self.los(time).with_deriv('los', Vector3.IDENTITY, 'insert')
        uv = self.fov.uv_from_los_t(los_, time=time, derivs=True,
                                    **self.fov_keywords)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self.filled_uv_w_derivs[key] = uv

        return uv

    #===========================================================================
    def duv_dlos(self, time=None):
        return self.uv_w_derivs(time).d_dlos

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Meshgrid(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
