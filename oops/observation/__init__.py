################################################################################
# oops/observation/__init__.py: Abstract class Observation
################################################################################

from __future__ import print_function

import numpy as np
import numbers

from polymath import Scalar, Pair, Vector, Vector3, Qube

from oops.config   import LOGGING, PATH_PHOTONS
from oops.event    import Event
from oops.meshgrid import Meshgrid

class Observation(object):
    """An Observation is an abstract class that defines the timing and pointing
    of the samples that comprise a data array.

    The axes of an observation are related to up to two spatial axes and one
    time axis. Spatial axes (u,v) are defined within an FOV (field of view)
    object. Time is specified as an offset in seconds relative to the start time
    of the observation. An observation provides methods to convert between the
    indices of the data array and the coordinates (u,v,t) that define a line of
    sight at a particular time.

    When indices have non-integer values, the integer part identifies one
    "corner" of the sample, and the fractional part locates a point within the
    sample, i.e., part way from the start time to the end time of an
    integration, or a location inside the boundaries of a spatial pixel.
    Half-integer indices falls at the midpoint of each sample.

    At minimum, these attributes are used to describe the observation:

        time            a tuple or Pair defining the start time and end time of
                        the observation overall, in seconds TDB.

        midtime         the mid-time of the observation, in seconds TDB.

        cadence         a Cadence object defining the timing of the observation.

        fov             a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y).

        uv_shape        a tuple defining the 2-D shape of the spatial axes of
                        the data array, in (u,v) order. Note that this may
                        differ from fov.uv_shape.

        u_axis, v_axis  integers identifying the axes of the data array
                        associated with the u-axis and the v-axis. Use -1 if
                        that axis is not associated with an array index.

        swap_uv         True if the v-axis comes before the u-axis;
                        False otherwise.

        t_axis          integers or lists of integers identifying the axes of
                        the data array associated with time. When a list has
                        multiple values, this is the sequence of array indices
                        that break down time into finer and finer divisions,
                        ordered from left to right. Use -1 if the observation
                        has no time-dependence.

        shape           a list or tuple defining the overall shape of the
                        observation data. Where the size of an axis is unknown
                        (e.g., for a wavelength axis), the value can be zero.

        path            the path waypoint co-located with the instrument.

        frame           the wayframe of a coordinate frame fixed to the optics
                        of the instrument. This frame should have its Z-axis
                        pointing outward near the center of the line of sight,
                        with the X-axis pointing rightward and the y-axis
                        pointing downward.

        subfields       a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
            data        a reserved subfield to contain the NumPy array of
                        numbers associated with the observation.
    """

    INVENTORY_IMPLEMENTED = False

    ############################################################################
    # Methods to be defined for each subclass
    ############################################################################

    def __init__(self):
        """A constructor."""

        pass

    #===========================================================================
    def uvt(self, indices, remask=False, derivs=True):
        """Coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Scalar or Vector of array indices.
            remask      True to mask values outside the field of view.
            derivs      True to include derivatives in the returned values.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) within the FOV that
                        are associated with the array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        raise NotImplementedError(type(self).__name__ + '.uvt ' +
                                  'is not implemented')

    #===========================================================================
    def uvt_range(self, indices, remask=False):
        """Ranges of (u,v) spatial coordinates and time for integer array
        indices.

        Input:
            indices     a Vector of array indices.
            remask      True to mask values outside the field of view.

        Return:         (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of (u,v) associated
                        the pixel.
            uv_max      a Pair defining the maximum values of (u,v).
            time_min    a Scalar defining the minimum time associated with the
                        pixel. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        raise NotImplementedError(type(self).__name__ + '.uvt_range ' +
                                  'is not implemented')

    #===========================================================================
    def time_range_at_uv(self, uv_pair, remask=False):
        """The start and stop times of the specified spatial pixel (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) data array coordinates,
                        truncated to integers if necessary.
            remask      True to mask values outside the field of view.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        raise NotImplementedError(type(self).__name__ + '.time_range_at_uv ' +
                                  'is not implemented')

    #===========================================================================
    def time_range_at_uv_0d(self, uv_pair, remask=False):
        """time_range_at_uv() for some observations in which the spatial and
        time axes are independent.

        Input:
            uv_pair     a Pair of spatial (u,v) data array coordinates,
                        truncated to integers if necessary.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        time_min = Scalar(self.time[0])     # shapeless scalars
        time_max = Scalar(self.time[1])

        if remask:
            uv_pair = Pair.as_pair(uv_pair, recursive=False)
            new_mask = self.fov.uv_is_outside(uv_pair)
            if new_mask.any_true_or_masked():
                new_mask = Qube.or_(new_mask.vals, new_mask.mask)
                time_min = Scalar.filled(uv_pair.shape, self.time[0],
                                                        mask=new_mask)
                time_max = Scalar.filled(uv_pair.shape, self.time[1],
                                                        mask=new_mask)

        return (time_min, time_max)

    #===========================================================================
    def time_range_at_uv_1d(self, uv_pair, axis=0, remask=False):
        """time_range_at_uv() for some observations with a 1-D cadence.

        Input:
            uv_pair     a Pair of spatial (u,v) data array coordinates,
                        truncated to integers if necessary.
            axis        0 or 1, indicating the uv axis associated with the
                        cadence.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        tstep = uv_pair.to_scalar(axis)

        # Re-mask the time-independent axis if necessary
        if remask:
            not_t_vals = uv_pair.vals[..., 1-axis]
            not_t_max = self.uv_shape[1-axis]
            new_mask = Qube.or_(not_t_vals < 0, not_t_vals > not_t_max)
            tstep = tstep.remask_or(new_mask)

        return self.cadence.time_range_at_tstep(tstep, remask=remask)

    #===========================================================================
    def time_range_at_uv_2d(self, uv_pair, fast=1, remask=False):
        """time_range_at_uv() for some observations with a 2-D cadence.

        Input:
            uv_pair     a Pair of spatial (u,v) data array coordinates,
                        truncated to integers if necessary.
            fast        0 or 1, indicating the uv axis associated with the
                        fast index of the cadence. The slow index is always
                        1 - fast.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=False)

        if fast == 1:
            return self.cadence.time_range_at_tstep(uv_pair, remask=remask)
        else:
            return self.cadence.time_range_at_tstep(uv_pair.swapxy(),
                                                    remask=remask)

    #===========================================================================
    def uv_range_at_time(self, time, remask=False):
        """The (u,v) range of spatial pixels in the data array observed at the
        specified time.

        Input:
            time        a Scalar of time values in seconds TDB.
            remask      True to mask values outside the time limits.

        Return:         (uv_min, uv_max)
            uv_min      the lower (u,v) corner Pair of the area observed at the
                        specified time.
            uv_max      the upper (u,v) corner Pair of the area observed at the
                        specified time.
        """

        raise NotImplementedError(type(self).__name__ + '.uv_range_at_time ' +
                                  'is not implemented')

    #===========================================================================
    def uv_range_at_time_0d(self, time, uv_shape, remask=False):
        """uv_range_at_time() for an observation in which any time-dependence is
        decoupled from the spatial axes.

        Input:
            time        time Scalar.
            uv_shape    shape of the active detector(s) within the FOV.
            remask      True to mask times that are out of range.
        """

        # Without re-masking, shapeless Pairs are OK
        if not remask:
            return (Pair.INT00, Pair.as_pair(uv_shape))

        # Define the new mask
        time = Scalar.as_scalar(time, derivs=False)
        new_mask = Qube.or_(time.mask, self.cadence.time_is_outside(time).vals)

        # Without any mask, shapeless Pairs are OK
        if not np.any(new_mask):
            return (Pair.INT00, Pair.as_pair(uv_shape))

        # Construct the array of results if necessary
        uv_min = Pair.zeros(time.shape, dtype='int', mask=new_mask)
        return (uv_min, uv_min + Pair.as_pair(uv_shape))

    #===========================================================================
    def uv_range_at_time_1d(self, time, uv_shape, axis=0, remask=False):
        """uv_range_at_time() for some observations with a 1-D cadence.

        Input:
            time        time Scalar.
            uv_shape    shape of the active detector(s) within the FOV.
            axis        0 or 1, indicating the uv axis associated with the
                        cadence. Alternatively, -1 indicates that time axis is
                        not associated with a spatial axis.
            remask      True to mask times that are out of range.
        """

        if axis < 0:
            return self.uv_range_at_time_0d(time, uv_shape, remask=remask)

        (tstep_min,
         tstep_max) = self.cadence.tstep_range_at_time(time, remask=remask)

        uv_min_vals = np.zeros(tstep_min.shape + (2,), dtype='int')
        uv_max_vals = np.empty(tstep_min.shape + (2,), dtype='int')

        uv_min_vals[..., axis] = tstep_min.vals
        uv_max_vals[..., axis] = tstep_max.vals
        uv_max_vals[..., 1-axis] = uv_shape[1-axis]

        uv_min = Pair(uv_min_vals, tstep_min.mask)
        uv_max = Pair(uv_max_vals, tstep_min.mask)
        return (uv_min, uv_max)

    #===========================================================================
    def uv_range_at_time_2d(self, time, uv_shape, slow=0, fast=1, remask=False):
        """uv_range_at_time() for some observations with a 2-D cadence.

        Input:
            time        time Scalar.
            uv_shape    shape of the active detector(s) within the FOV.
            slow, fast  0 or 1, indicating the uv axes associated with the slow
                        and fast indices of the cadence. Alternatively, -1
                        indicates that time axis is not associated with a
                        spatial axis.
            remask      True to mask times that are out of range.
        """

        (tstep_min,
         tstep_max) = self.cadence.tstep_range_at_time(time, remask=remask)

        if slow == 0 and fast == 1:
            return (tstep_min, tstep_max)
        elif slow == 1 and fast == 0:
            return (tstep_min.swapxy(), tstep_max.swapxy())

        uv_min_vals = np.zeros(tstep_min.shape + (2,), dtype='int')
        uv_max_vals = np.empty(tstep_min.shape + (2,), dtype='int')
        uv_max_vals[..., 0] = uv_shape[0]
        uv_max_vals[..., 1] = uv_shape[1]

        if slow >= 0:
            uv_min_vals[..., slow] = tstep_min.vals[..., 0]
            uv_max_vals[..., slow] = tstep_max.vals[..., 0]
        if fast >= 0:
            uv_min_vals[..., fast] = tstep_min.vals[..., 1]
            uv_max_vals[..., fast] = tstep_max.vals[..., 1]

        uv_min = Pair(uv_min_vals, tstep_min.mask)
        uv_max = Pair(uv_max_vals, tstep_min.mask)
        return (uv_min, uv_max)

    #===========================================================================
    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        raise NotImplementedError(type(self).__name__ + '.time_shift ' +
                                  'is not implemented')

    ############################################################################
    # Subfield support methods
    ############################################################################

    def insert_subfield(self, key, value):
        """Add a given subfield to the Event."""

        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

    #===========================================================================
    def delete_subfield(self, key):
        """Delete a subfield, but not arr or dep."""

        if key in self.subfields:
            del self.subfields[key]
            del self.__dict__[key]

    #===========================================================================
    def delete_subfields(self):
        """Delete all subfields."""

        for key in self.subfields:
            del self.subfields[key]
            del self.__dict__[key]

    ############################################################################
    # Methods probably not requiring overrides
    ############################################################################

    def uv_is_outside(self, uv_pair, inclusive=True):
        """A Boolean mask identifying coordinates outside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.

        Return:         a Boolean indicating True where the point is outside the
                        FOV.
        """

        # Interpret the (u,v) coordinates
        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        (u,v) = uv_pair.to_scalars()

        # Create the mask
        if inclusive:
            return (u.tvl_lt(0) | v.tvl_lt(0) | u.tvl_gt(self.uv_shape[0])
                                              | v.tvl_gt(self.uv_shape[1]))
        else:
            return (u.tvl_lt(0) | v.tvl_lt(0) | u.tvl_gt(self.uv_shape[0])
                                              | v.tvl_ge(self.uv_shape[1]))

    #===========================================================================
    def midtime_at_uv(self, uv, tfrac=0.5):
        """The mid-time for the selected spatial pixel (u,v).

        Input:
            uv          a Pair of (u,v) coordinates.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
        """

        (time0, time1) = self.time_range_at_uv(uv)
        return tfrac * (time0 + time1)

    #===========================================================================
    def meshgrid(self, origin=0.5, undersample=1, oversample=1, limit=None,
                       fov_keywords={}):
        """A Meshgrid shaped to broadcast to the observation's shape.

        This works like Meshgrid.for_fov() except that the (u,v) axes are
        assigned their correct locations in the axis ordering of the
        observation.

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

            fov_keywords  an optional dictionary of parameters passed to the
                        FOV methods, containing parameters that might affect
                        the properties of the FOV.
        """

        # Convert inputs to NumPy 2-element arrays
        if limit is None:
            limit = self.uv_shape
        if isinstance(limit, numbers.Number):
            limit = (limit, limit)
        limit = Pair.as_pair(limit).vals.astype('float')

        if isinstance(origin, numbers.Number):
            origin = (origin, origin)
        origin = Pair.as_pair(origin).vals.astype('float')

        if isinstance(undersample, numbers.Number):
            undersample = (undersample, undersample)
        undersample = Pair.as_pair(undersample).vals.astype('float')

        if isinstance(oversample, numbers.Number):
            oversample = (oversample, oversample)
        oversample = Pair.as_pair(oversample).vals.astype('float')

        # Construct the 1-D index arrays
        step = undersample/oversample
        limit = limit + step * 1.e-10   # Allow a little slop at the upper end

        urange = np.arange(origin[0], limit[0], step[0])
        vrange = np.arange(origin[1], limit[1], step[1])

        usize = urange.size
        vsize = vrange.size

        # Construct the empty array of values
        shape_list = len(self.shape) * [1]
        if self.u_axis >= 0:
            shape_list[self.u_axis] = usize
        if self.v_axis >= 0:
            shape_list[self.v_axis] = vsize

        values = np.empty(tuple(shape_list + [2]))

        # Populate the array
        if self.u_axis >= 0:
            shape_list = len(self.shape) * [1]
            shape_list[self.u_axis] = usize
            uvalues = urange.reshape(tuple(shape_list))
            values[...,0] = uvalues
        else:
            values[...,0] = 0.5

        if self.v_axis >= 0:
            shape_list = len(self.shape) * [1]
            shape_list[self.v_axis] = vsize
            vvalues = vrange.reshape(tuple(shape_list))
            values[...,1] = vvalues
        else:
            values[...,1] = 0.5

        # Return the Meshgrid
        grid = Pair(values)
        return Meshgrid(self.fov, grid, fov_keywords)

    #===========================================================================
    def timegrid(self, meshgrid, oversample=1, tfrac_limits=(0,1)):
        """A Scalar of times broadcastable with the shape of the given meshgrid.

        Input:
            meshgrid        the meshgrid defining spatial sampling.
            oversample      1 to obtain one time sample per pixel; > 1 for finer
                            sampling in time.

            tfrac_limits    a tuple interpreted in different ways depending on
                            the observation's structure.
                            - if this observation has no time-dependence, it is
                              the pair of fractional time limits within the
                              overall exposure duration.
                            - if this observation has time-dependence that is
                              entirely coupled to spatial axes, then it is the
                              fractional time limits within each pixel's
                              individual exposure duration.
                            - if this observation has time-dependence that is
                              entirely decoupled from the spatial axes, then it
                              is the start and end time relative to the time
                              limits of the defined cadence.
                            - the possible case of a 2-D time-dependence that
                              has only one axis coupled to a spatial axis is not
                              supported.
        """

        if isinstance(tfrac_limits, numbers.Number):
            tfrac_limits = (tfrac_limits, tfrac_limits)

        # Handle a time-independent observation
        if self.t_axis == -1:

            dt = self.time[1] - self.time[0]
            time0 = self.time[0] + tfrac_limits[0] * dt
            time1 = self.time[0] + tfrac_limits[1] * dt

            # One step implies midtime, which can be returned as a scalar
            if oversample == 1:
                return Scalar(0.5 * (time0 + time1))

            # Otherwise, uniform time steps between endpoints
            fracs = np.arange(oversample) / (oversample - 1.)
            times = time0 + fracs * (time1 - time0)

            # Time is on a leading axis
            tshape = times.shape + len(self.shape) * (1,)
            return Scalar.as_scalar(times.reshape(tshape))

        # Get times at each pixel in meshgrid
        (tstarts, tstops) = self.time_range_at_uv(meshgrid.uv)

        # Scale based on tfrac_limits
        time0 = tstarts + tfrac_limits[0] * (tstops - tstarts)
        time1 = tstarts + tfrac_limits[1] * (tstops - tstarts)

        # Handle 1-D case
        if isinstance(self.t_axis, numbers.Number):

            # Time aligns with u-axis or v-axis
            if self.t_axis in (self.u_axis, self.v_axis):

                # One time step implies midtime
                if oversample == 1:
                    return Scalar.as_scalar(0.5 * (time0 + time1))

                # Otherwise, uniform time steps on a leading axis
                fracs = np.arange(oversample) / (oversample - 1.)
                fracs = fracs.reshape(fracs.shape + len(self.shape) * (1,))
                return Scalar(time0 + fracs * (time1 - time0))

            # Otherwise time is along a unique axis
            tstep0 = tfrac_limits[0] * self.cadence.shape[0]
            tstep1 = tfrac_limits[1] * self.cadence.shape[0]
            tsteps = np.arange(tstep0, tstep1 + 1.e-10, 1./oversample)
            times = self.cadence.time_at_tstep(tsteps)

            shape_list = len(self.shape) * [1]
            shape_list[self.t_axis] = len(times)
            times = Scalar.as_scalar(times).reshape(tuple(shape_list))
            return times

        # Handle a 2-D observation
        if (self.t_axis[0] not in (self.u_axis, self.v_axis) or
            self.t_axis[1] not in (self.u_axis, self.v_axis)):
                raise NotImplementedError('Observation.timegrid not ' +
                                          'implemented for ' +
                                          't axes (%d,%d), ' % self.t_axis,
                                          'u axis %d, ' % self.u_axis,
                                          'v axis %d'   % self.v_axis)

        # Time aligns with u-axis AND v-axis

        # One time step implies midtime
        if oversample == 1:
            return Scalar.as_scalar(0.5 * (time0 + time1))

        # Otherwise, uniform time steps on a leading axis
        fracs = np.arange(oversample) / (oversample - 1.)
        fracs = fracs.reshape(fracs.shape + len(self.shape) * (1,))
        return Scalar(time0 + fracs * (time1 - time0))

    #===========================================================================
    def event_at_grid(self, meshgrid=None, tfrac=0.5, time=None):
        """A photon arrival event from directions defined by a meshgrid.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        optional Scalar of absolute time in seconds. Only one of
                        tfrac and time can be specified.

        Return:         the corresponding event.
        """

        if time is None:
            time = self.midtime_at_uv(meshgrid.uv, tfrac=tfrac)

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        # Insert the arrival directions
        event.neg_arr_ap = meshgrid.los(time)

        return event

    #===========================================================================
    def gridless_event(self, meshgrid=None, tfrac=0.5, time=None,
                             shapeless=False):
        """A photon arrival event irrespective of the direction.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view; None for a directionless observation. Here, it
                        is only used to define the times if time is None.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5. Ignored if time is specified.
            time        Scalar of optional absolute time in seconds.
            shapeless   True to return a shapeless event, referring to the mean
                        of all the times.

        Return:         the corresponding event.
        """

        if time is None:
            time = self.midtime_at_uv(meshgrid.uv, tfrac=tfrac)

        if shapeless:
            time = time.mean()

        return Event(time, Vector3.ZERO, self.path, self.frame)

    #===========================================================================
    @staticmethod
    def scalar_from_indices(indices, axis, derivs=True):
        """Utility to return the selected Scalar from a Scalar or Vector of
        indices, np.ndarray, or a number.
        """

        if axis < 0:
            return None

        if isinstance(indices, (Scalar, Pair, Vector)):
            return indices.to_scalar(axis, recursive=derivs)

        if isinstance(indices, numbers.Real):
            assert axis == 0
            return Scalar(indices)

        indices = np.array(indices)

        # The meaning of the last axis in a Numpy array is ambiguous
        if indices.shape[-1] > axis:
            return Scalar(indices[..., axis])

        return Scalar(indices)                  # might fail; not our problem

    ############################################################################
    # Geometry solvers
    ############################################################################

    def uv_from_ra_and_dec(self, ra, dec, tfrac=0.5, time=None, apparent=True,
                           derivs=False, iters=2, quick={}):
        """Convert arbitrary scalars of RA and dec to FOV (u,v) coordinates.

        Input:
            ra          a Scalar of J2000 right ascensions.
            dec         a Scalar of J2000 declinations.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified.
            apparent    True to interpret the (RA,dec) values as apparent
                        coordinates; False to interpret them as actual
                        coordinates. Default is True.
            derivs      True to propagate derivatives of ra and dec through to
                        derivatives of the returned (u,v) Pairs.
            iters       the number of iterations to perform until convergence
                        is reached. Two is the most that should ever be needed;
                        Snapshot should override to one.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

        Return:         a Pair of (u,v) coordinates.

        Note: The only reasons for iteration are that the C-matrix and the
        velocity WRT the SSB could vary during the observation. I doubt this
        would ever be significant.
        """

        # Convert given (ra,dec) to line of sight in SSB/J2000 frame
        neg_arr_j2000 = Vector3.from_ra_dec_length(ra, dec, recursive=derivs)

        # Interpret the time
        if time is None:
            obs_time = self.time[0] + tfrac * (self.time[1] - self.time[0])

            # Require extra at least two iterations if tfrac != 0.5
            if not (Scalar.as_scalar(Scalar.as_scalar(tfrac) == 0.5)).all():
                iters = max(2, iters)

        else:
            obs_time = time
            iters = 1

        # Iterate until (u,v) has converged
        uv = None
        for count in range(iters):

            # Define the photon arrival event
            obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)

            if apparent:
                obs_event.neg_arr_ap_j2000 = neg_arr_j2000
            else:
                obs_event.neg_arr_j2000 = neg_arr_j2000

            # Convert to FOV coordinates
            prev_uv = uv
            uv = self.fov.uv_from_los_t(obs_event.neg_arr_ap, time=obs_time,
                                        derivs=derivs)

            # If this is the last iteration, we're done
            if count + 1 == iters:
                break

            # Update the time
            (t0, t1) = self.time_range_at_uv(uv)
            obs_time = t0 + tfrac * (t1 - t0)

            # Stop at convergence
            if uv == prev_uv:
                break

        return uv

    #===========================================================================
    def uv_from_path(self, path, tfrac=0.5, time=None, derivs=False, guess=None,
                           quick={}, converge={}):
        """The (u,v) indices of an object in the FOV, given its path.
        **** NOT WELL TESTED! ****

        Note: This procedure assumes that movement along a path is very limited
        during the exposure time of an individual pixel. It could fail to
        converge if there is a large gap in timing between adjacent pixels at a
        time when the object is crossing that gap. However, even then, it should
        select roughly the correct location. It could also fail to converge
        during a fast slew.

        Input:
            path        a Path object.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      True to propagate derivatives of the link time and
                        position into the returned event.
            guess       an optional guess at the light travel time from the path
                        to the event.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         the (u,v) indices of the pixel in which the point was
                        found. The path is evaluated at the mid-time of this
                        pixel.
        """

        # Assemble convergence parameters
        if converge:
            defaults = PATH_PHOTONS.__dict__.copy()
            defaults.update(converge)
            converge = defaults
        else:
            converge = PATH_PHOTONS.__dict__

        iters = converge['max_iterations']
        dlt_precision = converge['dlt_precision']

        # Interpret the time
        if time is None:
            obs_time = self.time[0] + tfrac * (self.time[1] - self.time[0])
        else:
            obs_time = time
            iters = 0

        # Require extra at least two iterations if tfrac != 0.5
        if not (Scalar.as_scalar(tfrac) == 0.5).all():
            iters = max(2, iters)

        # Iterate to solution...
        guess = None
        max_dt = np.inf
        for count in range(iters):

            # Locate the object in the field of view
            obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)
            (path_event,
             obs_event) = path.photon_to_event(obs_event,
                                               derivs=False, guess=guess,
                                               quick=quick, converge=converge)
            guess = path_event.time
            (uv_min, uv_max) = self.uv_at_time(obs_event.time)

            # Update the observation times based on pixel midtimes
            (t0, t1) = self.time_range_at_uv(uv_min)
            new_obs_time = t0 + tfrac * (t1 - t0)

            # Test for convergence
            prev_max_dt = max_dt
            max_dt = abs(new_obs_time - obs_time).max()
            obs_time = new_obs_time

            if LOGGING.observation_iterations:
                print(LOGGING.prefix, 'Observation.uv_from_path', count+1,
                                      max_dt)

            if max_dt <= dlt_precision or max_dt >= prev_max_dt:
                break

        # Return the results
        obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)

        (path_event,
         obs_event) = path.photon_to_event(obs_event,
                                           derivs=derivs, guess=guess,
                                           quick=quick, converge=converge)

        return self.fov.uv_from_los_t(obs_event.neg_arr_ap, time=obs_time,
                                      derivs=derivs)

    #===========================================================================
    def uv_from_coords(self, surface, coords, tfrac=0.5, time=None,
                             underside=False, derivs=False,
                             quick={}, converge={}):
        """The (u,v) indices of a surface point, given its coordinates.

        Input:
            surface     a Surface object.
            coords      a tuple containing two or three Scalars of surface
                        coordinates. The Scalars need not be the same shape,
                        but must broadcast to the same shape.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            underside   True for the underside of the surface (emission > 90
                        degrees) to be unmasked.
            derivs      True to propagate derivatives of the link time and
                        position into the returned event.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         the (u,v) indices of the pixel in which the point was
                        found.
        """

        raise NotImplementedError(type(self).__name__ + '.uv_from_coords '
                                  'is not implemented')

    #===========================================================================
    def inventory(self, bodies, tfrac=0.5, time=None, expand=0.,
                        return_type='list', fov=None, quick={}, converge={}):
        """Info about the bodies that appear unobscured inside the FOV.

        Restrictions: All inventory calculations are performed at a single
        observation time specified by tfrac. All bodies are assumed to be
        spherical.

        Input:
            bodies      a list of the names of the body objects to be included
                        in the inventory.
            tfrac       fractional time from the beginning to the end of the
                        observation for which the inventory applies. 0 for the
                        beginning; 0.5 for the midtime, 1 for the end time.
                        Ignored if time is specified.
            time        Scalar of optional absolute time in seconds.
            expand      an optional angle in radians by which to extend the
                        limits of the field of view. This can be used to
                        accommodate pointing uncertainties. XXX NOT IMPLEMENTED XXX
            return_type 'list' returns the inventory as a list of names.
                        'flags' returns the inventory as an array of boolean
                                flag values in the same order as bodies.
                        'full' returns the inventory as a dictionary of
                                dictionaries. The main dictionary is indexed by
                                body name. The subdictionaries contain
                                attributes of the body in the FOV.
            fov         use this fov; if None, use self.fov.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.

        Return:         list, array, or dictionary

            If return_type is 'list', it returns a list of the names of all the
            body objects that fall at least partially inside the FOV and are
            not completely obscured by another object in the list.

            If return_type is 'flags', it returns a boolean array containing
            True everywhere that the body falls at least partially inside the
            FOV and is not completely obscured.

            If return_type is 'full', it returns a dictionary with one entry
            per body that falls at least partially inside the FOV and is not
            completely obscured. Each dictionary entry is itself a dictionary
            containing data about the body in the FOV:

                body_data['name']          The body name
                body_data['center_uv']     The U,V coord of the center point
                body_data['center']        The Vector3 direction of the center
                                           point
                body_data['range']         The range in km
                body_data['outer_radius']  The outer radius of the body in km
                body_data['inner_radius']  The inner radius of the body in km
                body_data['resolution']    The resolution (km/pix) in the (U,V)
                                           directions at the given range.
                body_data['u_min']         The minimum U value covered by the
                                           body (clipped to the FOV size)
                body_data['u_max']         The maximum U value covered by the
                                           body (clipped to the FOV size)
                body_data['v_min']         The minimum V value covered by the
                                           body (clipped to the FOV size)
                body_data['v_max']         The maximum V value covered by the
                                           body (clipped to the FOV size)
                body_data['u_min_unclipped']  Same as above, but not clipped
                body_data['u_max_unclipped']  to the FOV size.
                body_data['v_min_unclipped']
                body_data['v_max_unclipped']
                body_data['u_pixel_size']  The number of pixels (non-integer)
                body_data['v_pixel_size']  covered by the diameter of the body
                                           in each direction.
        """

        raise NotImplementedError(type(self).__name__ + '.inventory '
                                  'is not implemented')

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Observation(unittest.TestCase):

    def runTest(self):

        # TBD
        # Note in particular that uv_from_path() is incomplete and untested!

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
