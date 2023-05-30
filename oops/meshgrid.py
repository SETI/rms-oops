################################################################################
# oops/meshgrid.py: Class Meshgrid
################################################################################

import numpy as np
import numbers

from polymath import Scalar, Pair, Vector3

class Meshgrid(object):
    """A Meshgrid is an arbitrary array of coordinate pairs within a Field of
    View. It caches information about the line of sight and various derivatives,
    preventing the need for repeated calls to the FOV functions when the same
    field of view describes multiple images.

    It has these key attributes:
        uv              the (u,v) meshgrid for indexing the spatial dimensions
                        of an observation's data array, yielding (u,v)
                        coordinates in the FOV.
        duv_dxy         the partial derivatives d(u,v)/d(x,y), where (x,y) are
                        two components of a unit line-of-sight vector (x,y,z).
        uv_w_derivs     the (u,v) pairs with d_dxy, where (x,y) are two
                        components of a unit line-of-sight vector (x,y,z).

        los             the unit line-of-sight vectors with no derivatives.
        dlos_duv        the partial derivatives dlos/d(u,v).
        los_w_derivs    the unit line-of-sight vectors with d_duv.

        shape           the (u,v) shape of the observation's data array.
    """

    #===========================================================================
    def __init__(self, fov, uv_pair, center_uv=None, fov_keywords={}):
        """The Meshgrid constructor.

        Input:
            fov             a FOV object.
            uv_pair         a Pair object of arbitrary shape, representing (u,v)
                            coordinates within a field of view.
            center_uv       (u,v) coordinates of the center of the meshgrid;
                            default is the mean of all the uv_pair values.
            fov_keywords    an optional dictionary of parameters passed to the
                            FOV methods, containing parameters that might affect
                            the properties of the FOV.
        """

        self.fov = fov
        self.uv = Pair.as_pair(uv_pair).wod
        self.uv_w_duv_duv = self.uv.with_deriv('uv', Pair.IDENTITY, 'insert')
        self.fov_keywords = fov_keywords
        self.shape = self.uv.shape

        # Cache for holding info if time is a single value or None
        self.filled_los_w_derivs = {}
        self.filled_los = {}
        self.filled_uv_w_derivs = {}

        # Center point
        if center_uv is None:
            self.center_uv = self.uv.mean()
        else:
            self.center_uv = Pair.as_pair(center_uv).wod

        self.center_uv_w_duv_duv = self.center_uv.with_deriv('uv',
                                                        Pair.IDENTITY, 'insert')

        self.filled_center_los_w_derivs = {}
        self.filled_center_los = {}
        self.filled_center_uv_w_derivs = {}

    def __getstate__(self):
        return (self.fov, self.uv, self.fov_keywords)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def for_fov(fov, origin=None, undersample=1, oversample=1, limit=None,
                     swap=False, fov_keywords={}):
        """A 2-D rectangular Meshgrid object for a specified sampling of the
        FOV.

        Input:
            fov         FOV object.

            origin      A single value, tuple or Pair defining the origin of the
                        grid. Default is to place the first sample in the middle
                        of the first pixel, allowing for under- or oversampling.

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

        u_axis, v_axis = (1,0) if swap else (0,1)
        return Meshgrid.for_shape(fov, fov.uv_shape.vals, u_axis, v_axis,
                                  origin=origin,
                                  undersample=undersample,
                                  oversample=oversample,
                                  limit=limit,
                                  fov_keywords=fov_keywords)

    #===========================================================================
    @staticmethod
    def for_fov_center(fov, origin=None, fov_keywords={}):
        """A 0-D Meshgrid object for a single line of sight within an FOV.

        Input:
            fov         FOV object.

            origin      A single value, tuple or Pair defining the line of sight
                        of the "grid". Default is to use the center of the FOV.

            fov_keywords  an optional dictionary of parameters passed to the
                        FOV methods, containing parameters that might affect
                        the properties of the FOV.
        """

        return Meshgrid(fov, fov.uv_shape/2., fov_keywords=fov_keywords)

    #===========================================================================
    @staticmethod
    def for_shape(fov, shape, u_axis=-1, v_axis=-1, origin=None, undersample=1,
                  oversample=1, limit=None, center_uv=None, fov_keywords={}):
        """A 2-D rectangular Meshgrid object for a specified FOV and uv_shape.

        Input:
            fov         FOV object.

            shape       overall shape to which this Meshgrid must broadcast.

            u_axis      location of the u axis within the shape; -1 if there is
                        no u-axis.

            v_axis      location of the v axis within the shape; -1 if there is
                        no v-axis.

            origin      A single value, tuple or Pair defining the (u,v) origin
                        of the grid. Default is to place the first sample in the
                        middle of the first pixel (after allowing for the under-
                        or oversampling).

            undersample A single value, tuple or Pair defining the magnitude of
                        under-sampling to be performed. For example, a value of
                        2 would cause the meshgrid to sample every other pixel
                        along each axis.

            oversample  A single value, tuple or Pair defining the magnitude of
                        over-sampling to be performed. For example, a value of
                        2 would create a 2x2 array of samples inside each pixel.

            limit       A single value, tuple or Pair defining the (u,v) upper
                        limits of the meshgrid. By default, this is the shape of
                        the FOV.

            center_uv   A single value, tuple or Pair defining the (u,v) center
                        of the FOV. Default is to place this point at the center
                        of the specified grid of points.

            fov_keywords  an optional dictionary of parameters passed to the
                        FOV methods, containing parameters that might affect
                        the properties of the FOV.
        """

        u_size = 1 if u_axis < 0 else shape[u_axis]
        v_size = 1 if v_axis < 0 else shape[v_axis]
        uv_shape = (u_size, v_size)

        # Locate the default (u,v) center
        if center_uv is None and origin is None and limit is None:
            center_uv = Pair.as_pair(uv_shape).vals / 2.

        # Convert inputs to NumPy 2-element arrays
        if limit is None:
            limit = uv_shape
        if isinstance(limit, numbers.Real):
            limit = (limit, limit)
        limit = Pair.as_pair(limit).as_float().vals

        if isinstance(undersample, numbers.Real):
            undersample = (undersample, undersample)
        undersample = Pair.as_pair(undersample).as_float().vals

        if isinstance(oversample, numbers.Real):
            oversample = (oversample, oversample)
        oversample = Pair.as_pair(oversample).as_float().vals

        # Valid value checks
        if np.any(undersample < 1):
            raise ValueError('invalid undersample: ' + repr(undersample))

        if np.any(oversample < 1):
            raise ValueError('invalid oversample: ' + repr(undersample))

        if np.any(np.minimum(undersample, oversample) != 1):
            raise ValueError('undersample and oversample cannot both be != 1')

        step = undersample/oversample

        if origin is None:
            origin = step / 2.
        if isinstance(origin, numbers.Real):
            origin = (origin, origin)
        origin = Pair.as_pair(origin).as_float().vals

        # Determine reference center point
        if center_uv is None:
            center_uv = 0.5 * (origin - step/2. + limit)

        # Construct the 1-D index arrays
        u_range = np.arange(origin[0], limit[0] + step[0]/1.e10, step[0])
        v_range = np.arange(origin[1], limit[1] + step[1]/1.e10, step[1])
            # We add a small amount to each upper limit just to avoid the
            # possible loss of the last step along each axis due to rounding
            # error.

        # Construct the array of (u,v) coordinates
        uv_vals = np.empty((len(u_range), len(v_range), 2))
        if u_axis >= 0:
            uv_vals[:,:,0] = u_range[:,np.newaxis]
        else:
            uv_vals[:,:,0] = 0.5

        if v_axis >= 0:
            uv_vals[:,:,1] = v_range
        else:
            uv_vals[:,:,1] = 0.5

        # Convert to the required shape
        shape_list = len(shape) * [1]
        if u_axis >= 0:
            shape_list[u_axis] = len(u_range)
        if v_axis >= 0:
            shape_list[v_axis] = len(v_range)

        if u_axis >= 0 and v_axis >= 0 and v_axis < u_axis:
            uv_vals = uv_vals.swapaxes(0,1)

        uv_vals = uv_vals.reshape(shape_list + [2])

        # Return the Meshgrid
        uv_pair = Pair(uv_vals)
        return Meshgrid(fov, uv_pair, center_uv=center_uv,
                                      fov_keywords=fov_keywords)

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
        result = self.fov.los_from_uvt(self.uv_w_duv_duv, time=time,
                                       derivs=True, **self.fov_keywords)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self.filled_los_w_derivs[key] = result

        return result

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
        result = self.fov.los_from_uvt(self.uv, time=time, derivs=False,
                                        **self.fov_keywords)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self.filled_los[key] = result

        return result

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

    ############################################################################
    # Center methods
    ############################################################################

    def center_los_w_derivs(self, time=None):

        # Return from internal dictionary if present
        try:
            return self.filled_center_los_w_derivs[time]
        except (KeyError, TypeError):
            pass

        # Evaluate the LOS anew
        los_ = self.fov.los_from_uvt(self.center_uv_w_duv_duv, time=time,
                                     derivs=True, **self.fov_keywords)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self.filled_center_los_w_derivs[key] = los_

        return los_

    #===========================================================================
    def center_los(self, time=None):

        # Return from internal dictionary if present
        # If it's in filled_center_los_w_derivs, adapt that one
        try:
            return self.filled_center_los[time]
        except KeyError:        # on a KeyError, we can try the other dict
            try:
                los_wod = self.filled_center_los_w_derivs[time].wod
            except KeyError:    # not found, so give up with dicts
                pass
            else:               # found, so strip the derivs, save and return
                self.filled_center_los[time] = los_wod
                return los_wod
        except TypeError:
            pass

        # Evaluate the LOS anew
        los_ = self.fov.los_from_uvt(self.center_uv, time=time, derivs=False,
                                     **self.fov_keywords)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self.filled_center_los[key] = los_

        return los_

    #===========================================================================
    def center_dlos_duv(self, time=None):
        return self.center_los_w_derivs(time).d_duv

    #===========================================================================
    def center_uv_w_derivs(self, time=None):

        # Return from internal dictionary if present
        try:
            return self.filled_center_uv_w_derivs[time]
        except (KeyError, TypeError):
            pass

        # Evaluate (u,v) anew
        los_ = self.center_los(time).with_deriv('los', Vector3.IDENTITY,
                                                'insert')
        uv = self.fov.uv_from_los_t(los_, time=time, derivs=True,
                                    **self.fov_keywords)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self.filled_center_uv_w_derivs[key] = uv

        return uv

    #===========================================================================
    def center_duv_dlos(self, time=None):
        return self.center_uv_w_derivs(time).d_dlos

################################################################################
# UNIT TESTS
################################################################################

# Tested by other modules, but here's a placeholder if we want to add more tests
#
# import unittest
#
# class Test_Meshgrid(unittest.TestCase):
#
#     def runTest(self):
#
#         # TBD
#         pass
#
# ########################################
# if __name__ == '__main__':
#     unittest.main(verbosity=2)
################################################################################
