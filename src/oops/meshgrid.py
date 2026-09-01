##########################################################################################
# oops/meshgrid.py
##########################################################################################

import numbers

import numpy as np

from polymath import Scalar, Pair, Vector3


class Meshgrid(object):
    """An arbitrary array of coordinate pairs within a Field of View.

    A Meshgrid caches information about the line of sight and various derivatives,
    preventing the need for repeated calls to the FOV functions when the same field of
    view describes multiple images.

    Properties:
        * fov (FOV): The field of view that this Meshgrid samples.
        * uv (Pair): The `(u,v)` coordinates of the samples, without derivatives.
        * uv_w_duv_duv (Pair): The same `(u,v)` coordinates, carrying the identity
          derivative `d(u,v)/d(u,v)`.
        * center_uv (Pair): The `(u,v)` coordinates of the center of the meshgrid.
        * center_uv_w_duv_duv (Pair): The center coordinates, carrying the identity
          derivative `d(u,v)/d(u,v)`.
        * shape (tuple[int, ...]): The shape of the array of `(u,v)` coordinates.

    Lines of sight and their derivatives are obtained from methods rather than
    attributes, because each one depends on the time at which the FOV is sampled. Every
    result is cached, so repeated calls at the same time are inexpensive.
    """

    def __init__(self, fov, uv_pair, center_uv=None, *, fov_kwargs=None):
        """The Meshgrid constructor.

        Parameters:
            fov (FOV): The associated FOV object.
            uv_pair (Pair): Object of arbitrary shape, representing (u,v) coordinates
                within a field of view.
            center_uv (Pair, optional): `(u,v)` coordinates of the center of the meshgrid;
                default is the mean of all the `uv_pair` values.
            fov_kwargs (dict, optional): Parameters passed to the FOV methods,
                containing parameters that might affect the properties of the FOV.
        """

        self.fov = fov
        self.uv = Pair.as_pair(uv_pair).wod
        self.uv_w_duv_duv = self.uv.with_deriv('uv', Pair.IDENTITY, method='insert')
        self._fov_kwargs = fov_kwargs or {}
        self.shape = self.uv.shape

        # Cache for holding info if time is a single value or None
        self._filled_los_w_derivs = {}
        self._filled_los = {}
        self._filled_uv_w_derivs = {}

        # Center point
        if center_uv is None:
            self.center_uv = self.uv.mean()
        else:
            self.center_uv = Pair.as_pair(center_uv).wod

        self.center_uv_w_duv_duv = self.center_uv.with_deriv('uv', Pair.IDENTITY,
                                                             method='insert')

        self._filled_center_los_w_derivs = {}
        self._filled_center_los = {}
        self._filled_center_uv_w_derivs = {}

    def __getstate__(self):
        return (self.fov, self.uv, self._fov_kwargs)

    def __setstate__(self, state):
        self.__init__(*state[:-1], fov_kwargs=state[-1])

    @staticmethod
    def for_fov(fov, origin=None, undersample=1, oversample=1, limit=None, swap=False,
                fov_kwargs=None):
        """A 2-D rectangular Meshgrid object for a specified sampling of the FOV.

        Parameters:
            fov (FOV): FOV object.
            origin (Pair, optional): A single value, tuple or Pair defining the origin of
                the grid. Default is to place the first sample in the middle of the first
                pixel, allowing for under- or oversampling.
            undersample (Pair, optional): A single value, tuple or Pair defining the
                magnitude of under-sampling to be performed. For example, a value of 2
                would cause the meshgrid to sample every other pixel along each axis.
            oversample (Pair, optional): A single value, tuple or Pair defining the
                magnitude of over-sampling to be performed. For example, a value of 2
                would create a 2x2 array of samples inside each pixel.
            limit (Pair, optional): A single value, tuple or Pair defining the upper
                limits of the meshgrid. By default, this is the shape of the FOV.
            swap (bool, optional): True to swap the order of the indices in the meshgrid,
                (v,u) instead of (u,v).
            fov_kwargs (dict, optional): Parameters passed to the FOV methods,
                containing parameters that might affect the properties of the FOV.

        Returns:
            Meshgrid: A Meshgrid sampling the entire field of view.

        Raises:
            ValueError: If `undersample` or `oversample` is less than one, or if both
                differ from one.
        """

        u_axis, v_axis = (1,0) if swap else (0,1)
        return Meshgrid.for_shape(fov, fov.uv_shape.vals, u_axis, v_axis,
                                  origin = origin,
                                  undersample = undersample,
                                  oversample = oversample,
                                  limit = limit,
                                  fov_kwargs = fov_kwargs)

    @staticmethod
    def for_fov_center(fov, origin=None, fov_kwargs=None):
        """A 0-D Meshgrid object for a single line of sight within an FOV.

        Parameters:
            fov (FOV): FOV object.
            origin (Pair, optional): A single value, tuple, or Pair defining the line of
                sight of the "grid". Default is to use the center of the FOV.
            fov_kwargs (dict, optional): Parameters passed to the FOV methods,
                containing parameters that might affect the properties of the FOV.

        Returns:
            Meshgrid: A shapeless Meshgrid containing a single line of sight.
        """

        if origin is None:
            origin = fov.uv_shape/2.

        return Meshgrid(fov, origin, fov_kwargs=fov_kwargs)

    @staticmethod
    def for_shape(fov, shape, u_axis=-1, v_axis=-1, origin=None, undersample=1,
                  oversample=1, limit=None, center_uv=None, fov_kwargs=None):
        """A 2-D rectangular Meshgrid object for a specified FOV and uv_shape.

        Parameters:
            fov (FOV): FOV object.
            shape (tuple): Overall shape to which this Meshgrid must broadcast.
            u_axis (int, optional): Location of the u axis within the shape; -1 if there
                is no u-axis, in which case the meshgrid has a single sample along u.
            v_axis (int, optional): Location of the v axis within the shape; -1 if there
                is no v-axis, in which case the meshgrid has a single sample along v.
            origin (Pair, optional): A single value, tuple or Pair defining the `(u,v)`
                origin of the grid. Default is to place the first sample in the middle of
                the first pixel (after allowing for the under- or oversampling).
            undersample (Pair, optional): A single value, tuple or Pair defining the
                magnitude of under-sampling to be performed. For example, a value of 2
                would cause the meshgrid to sample every other pixel along each axis.
            oversample (Pair, optional): A single value, tuple or Pair defining the
                magnitude of over-sampling to be performed. For example, a value of 2
                would create a 2x2 array of samples inside each pixel.
            limit (Pair, optional): A single value, tuple or Pair defining the `(u,v)`
                upper limits of the meshgrid. By default, this is the shape of the FOV.
            center_uv (optional): A single value, tuple or Pair defining the `(u,v)`
                center of the FOV. Default is to place this point at the center of the
                specified grid of points.
            fov_kwargs (dict, optional): Parameters passed to the FOV methods,
                containing parameters that might affect the properties of the FOV.

        Returns:
            Meshgrid: A Meshgrid broadcastable to the given shape.

        Raises:
            ValueError: If `undersample` or `oversample` is less than one, or if both
                differ from one.
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
            raise ValueError('invalid oversample: ' + repr(oversample))

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

        # Construct the 1-D index arrays. An axis that is absent from the shape gets a
        # single sample, because the shape reserves room for only one; its coordinate is
        # filled in below. Sampling it as if it were present would leave more values than
        # the final reshape can hold.
        if u_axis >= 0:
            u_range = np.arange(origin[0], limit[0] + step[0]/1.e10, step[0])
        else:
            u_range = np.zeros(1)

        if v_axis >= 0:
            v_range = np.arange(origin[1], limit[1] + step[1]/1.e10, step[1])
        else:
            v_range = np.zeros(1)
            # We add a small amount to each upper limit just to avoid the possible loss
            # of the last step along each axis due to rounding error.

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
        return Meshgrid(fov, uv_pair, center_uv=center_uv, fov_kwargs=fov_kwargs)

    @staticmethod
    def _as_key(time):
        """The given time converted to a key for the internal caches.

        Parameters:
            time (Scalar, float, or None): Absolute time in seconds TDB.

        Returns:
            float, None, or bool: The time in a hashable form, or False if this time
            cannot be used as a cache key.
        """

        if time is None:
            return time

        if isinstance(time, numbers.Real):
            return time

        if isinstance(time, Scalar) and np.isscalar(time.vals):
            return time.vals

        return False

    def los_w_derivs(self, time=None):
        """The unit lines of sight, with derivatives with respect to `(u,v)`.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Vector3: The lines of sight, carrying the derivative `d_duv`.
        """

        # Return from internal dictionary if present
        try:
            return self._filled_los_w_derivs[time]
        except (KeyError, TypeError):
            pass

        # Evaluate the LOS anew
        result = self.fov.los_from_uvt(self.uv_w_duv_duv, time=time, derivs=True,
                                       **self._fov_kwargs)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self._filled_los_w_derivs[key] = result

        return result

    def los(self, time=None):
        """The unit lines of sight, without derivatives.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Vector3: The lines of sight at each sample of the meshgrid.
        """

        # Return from internal dictionary if present
        # If it's in _filled_los_w_derivs, adapt that one
        try:
            return self._filled_los[time]
        except KeyError:        # on a KeyError, we can try the other dict
            try:
                los_wod = self._filled_los_w_derivs[time].wod
            except KeyError:    # not found, so give up with dicts
                pass
            else:               # found, so strip the derivs, save and return
                self._filled_los[time] = los_wod
                return los_wod
        except TypeError:
            pass

        # Evaluate the LOS anew
        result = self.fov.los_from_uvt(self.uv, time=time, derivs=False,
                                        **self._fov_kwargs)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self._filled_los[key] = result

        return result

    def dlos_duv(self, time=None):
        """The partial derivatives of the lines of sight with respect to `(u,v)`.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Vector3: The derivatives `dlos/d(u,v)`, with `(u,v)` as the denominator.
        """
        return self.los_w_derivs(time).d_duv

    def uv_w_derivs(self, time=None):
        """The `(u,v)` coordinates, with derivatives with respect to the line of sight.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Pair: The `(u,v)` coordinates, carrying the derivative `d_dlos`.
        """

        # Return from internal dictionary if present
        try:
            return self._filled_uv_w_derivs[time]
        except (KeyError, TypeError):
            pass

        # Evaluate (u,v) anew
        los_ = self.los(time).with_deriv('los', Vector3.IDENTITY, method='insert')
        uv = self.fov.uv_from_los_t(los_, time=time, derivs=True, **self._fov_kwargs)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self._filled_uv_w_derivs[key] = uv

        return uv

    def duv_dlos(self, time=None):
        """The partial derivatives of the `(u,v)` coordinates with respect to the line of
        sight.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Pair: The derivatives `d(u,v)/dlos`, with the line of sight as the
            denominator.
        """
        return self.uv_w_derivs(time).d_dlos

    ######################################################################################
    # Center methods
    ######################################################################################

    def center_los_w_derivs(self, time=None):
        """The unit line of sight at the center, with derivatives with respect to `(u,v)`.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Vector3: The central line of sight, carrying the derivative `d_duv`.
        """

        # Return from internal dictionary if present
        try:
            return self._filled_center_los_w_derivs[time]
        except (KeyError, TypeError):
            pass

        # Evaluate the LOS anew
        los_ = self.fov.los_from_uvt(self.center_uv_w_duv_duv, time=time, derivs=True,
                                     **self._fov_kwargs)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self._filled_center_los_w_derivs[key] = los_

        return los_

    def center_los(self, time=None):
        """The unit line of sight at the center of the meshgrid, without derivatives.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Vector3: The line of sight at the center of the meshgrid.
        """

        # Return from internal dictionary if present
        # If it's in _filled_center_los_w_derivs, adapt that one
        try:
            return self._filled_center_los[time]
        except KeyError:        # on a KeyError, we can try the other dict
            try:
                los_wod = self._filled_center_los_w_derivs[time].wod
            except KeyError:    # not found, so give up with dicts
                pass
            else:               # found, so strip the derivs, save and return
                self._filled_center_los[time] = los_wod
                return los_wod
        except TypeError:
            pass

        # Evaluate the LOS anew
        los_ = self.fov.los_from_uvt(self.center_uv, time=time, derivs=False,
                                     **self._fov_kwargs)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self._filled_center_los[key] = los_

        return los_

    def center_dlos_duv(self, time=None):
        """The partial derivatives of the central line of sight with respect to `(u,v)`.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Vector3: The derivatives `dlos/d(u,v)` at the center, with `(u,v)` as the
            denominator.
        """
        return self.center_los_w_derivs(time).d_duv

    def center_uv_w_derivs(self, time=None):
        """The central `(u,v)` coordinates, with derivatives with respect to the line of
        sight.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Pair: The central `(u,v)` coordinates, carrying the derivative `d_dlos`.
        """

        # Return from internal dictionary if present
        try:
            return self._filled_center_uv_w_derivs[time]
        except (KeyError, TypeError):
            pass

        # Evaluate (u,v) anew
        los_ = self.center_los(time).with_deriv('los', Vector3.IDENTITY, method='insert')
        uv = self.fov.uv_from_los_t(los_, time=time, derivs=True, **self._fov_kwargs)

        # Save it in the dictionary if possible
        key = Meshgrid._as_key(time)
        if key is not False:
            self._filled_center_uv_w_derivs[key] = uv

        return uv

    def center_duv_dlos(self, time=None):
        """The partial derivatives of the central `(u,v)` coordinates with respect to the
        line of sight.

        Parameters:
            time (Scalar, optional): Absolute time in seconds TDB; None if the FOV does
                not depend on time.

        Returns:
            Pair: The derivatives `d(u,v)/dlos` at the center, with the line of sight
            as the denominator.
        """
        return self.center_uv_w_derivs(time).d_dlos

##########################################################################################
