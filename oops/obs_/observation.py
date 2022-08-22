################################################################################
# oops/obs_/observation.py: Abstract class Observation
################################################################################

from __future__ import print_function

import numpy as np
import numbers
from polymath import *

from oops.config            import LOGGING, PATH_PHOTONS
from oops.event             import Event
from oops.frame_.frame      import Frame
from oops.meshgrid          import Meshgrid
from oops.body              import Body
from oops.cadence_.cadence  import Cadence

#*******************************************************************************
# Observation
#*******************************************************************************
class Observation(object):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    An Observation is an abstract class that defines the timing and pointing
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
        uv_shape        a list or tuple defining the 2-D shape of the data array
                        in (u,v) order. Note that this may differ from
                        fov.uv_shape.
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
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    INVENTORY_IMPLEMENTED = False

    ####################################################
    # Methods to be defined for each subclass
    ####################################################

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A constructor.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pass
    #===========================================================================



    #===========================================================================
    # uvt
    #===========================================================================
    def uvt(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Tuple of array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raise NotImplementedError("uvt() is not implemented")
    #===========================================================================



    #===========================================================================
    # uvt_range
    #===========================================================================
    def uvt_range(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return ranges of coordinates and time for integer array indices.

        Input:
            indices     a Tuple of integer array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of (u,v) associated
                        the pixel.
            uv_max      a Pair defining the maximum values of (u,v).
            time_min    a Scalar defining the minimum time associated with the
                        pixel. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raise NotImplementedError("uvt_range() is not implemented")
    #===========================================================================



    #===========================================================================
    # uvt_ranges
    #===========================================================================
    def uvt_ranges(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a list of tuples defining the ranges of FOV coordinates and
        time limits at which they are applicable.

        Return:         a list of tuples (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of (u,v) associated
                        a tile of the FOV.
            uv_max      a Pair defining the maximum values of (u,v) associated
                        a tile of the FOV.
            time_min    a Scalar defining the minimum time associated with this
                        tile. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time associated with this
                        tile. It is given in seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raise NotImplementedError("uvt_ranges() is not implemented")
    #===========================================================================



    #===========================================================================
    # indices_at_uvt
    #===========================================================================
    def indices_at_uvt(self, uv_pair, time, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a vector of indices for given FOV coordinates (u,v) and time.

        This method supports non-integer positions and time steps, and returns
        fractional indices.

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in or near the field
                        of view.
            time        a Scalar of times in seconds TDB.
            fovmask     True to mask values outside the field of view.

        Return:
            indices     a Tuple of array indices. Any array indices not
                        constrained by (u,v) or time are returned with value 0.
                        Note that returned indices can fall outside the nominal
                        limits of the data object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raise NotImplementedError("indices_at_uvt() is not implemented")
    #===========================================================================



    #===========================================================================
    # times_at_uv
    #===========================================================================
    def times_at_uv(self, uv_pair, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return start and stop times of the specified spatial pixel (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raise NotImplementedError("times_at_uv() is not implemented")
    #===========================================================================



    #===========================================================================
    # uv_at_time
    #===========================================================================
    def uv_at_time(self, time, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The (u,v) range of spatial pixels observed at the specified time.

        Input:
            time        a Scalar of time values in seconds TDB.
            tmask       True to mask values outside the time limits.

        Return:         (uv_min, uv_max)
            uv_min      the lower (u,v) corner of the area observed at the
                        specified time.
            uv_max      the upper (u,v) corner of the area observed at the
                        specified time.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raise NotImplementedError("uv_at_time() is not implemented")
    #===========================================================================



    #===========================================================================
    # time_shift
    #===========================================================================
    def time_shift(self, dtime):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raise NotImplementedError("time_shift() is not implemented")
    #===========================================================================



    ####################################################
    # Subfield support methods
    ####################################################

    #===========================================================================
    # insert_subfield
    #===========================================================================
    def insert_subfield(self, key, value):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Adds a given subfield to the Event.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well
    #===========================================================================



    #===========================================================================
    # delete_subfield
    #===========================================================================
    def delete_subfield(self, key):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Deletes a subfield, but not arr or dep.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if key in self.subfields:
            del self.subfields[key]
            del self.__dict__[key]
    #===========================================================================



    #===========================================================================
    # delete_subfields
    #===========================================================================
    def delete_subfields(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Deletes all subfields.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key in self.subfields:
            del self.subfields[key]
            del self.__dict__[key]
    #===========================================================================



    ####################################################
    # Methods probably not requiring overrides
    ####################################################

    #===========================================================================
    # uv_is_outside
    #===========================================================================
    def uv_is_outside(self, uv_pair, inclusive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a boolean mask identifying coordinates outside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.

        Return:         a boolean NumPy array indicating True where the point is
                        outside the FOV.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#       This was wrong, because the (u,v) dimensions of the FOV do not
#       necessarily match the (u,v) dimensions of the observation.
#         return self.fov.uv_is_outside(uv_pair, inclusive)

        #-----------------------------------------------------
        # Interpret the (u,v) coordinates
        #-----------------------------------------------------
        uv_pair = Pair.as_pair(uv_pair)
        (u,v) = uv_pair.to_scalars()

        #-----------------------------------------------------
        # Create the mask
        #-----------------------------------------------------
        if inclusive:
            result = ((u < 0) | (v < 0) | (u > self.uv_shape[0])
                                        | (v > self.uv_shape[1]))
        else:
            result = ((u < 0) | (v < 0) | (u >= self.uv_shape[0])
                                        | (v >= self.uv_shape[1]))

        #-----------------------------------------------------
        # Convert to a boolean mask if necessary
        #-----------------------------------------------------
        if isinstance(result, Qube):
            return result.values        # Convert to NumPy
        else:
            return result               # bool
    #===========================================================================



    #===========================================================================
    # midtime_at_uv
    #===========================================================================
    def midtime_at_uv(self, uv):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the mid-time for the selected spatial pixel (u,v).

        Input:
            uv          a Pair of (u,v) coordinates.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        (time0, time1) = self.times_at_uv(uv)
        return 0.5 * (time0 + time1)
    #===========================================================================



    #===========================================================================
    # meshgrid
    #===========================================================================
    def meshgrid(self, origin=0.5, undersample=1, oversample=1, limit=None,
                       fov_keywords={}):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a Meshgrid shaped to broadcast to the observation's shape.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #---------------------------------------------
        # Convert inputs to NumPy 2-element arrays
        #---------------------------------------------
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

        #-----------------------------------
        # Construct the 1-D index arrays
        #-----------------------------------
        step = undersample/oversample
        limit = limit + step * 1.e-10   # Allow a little slop at the upper end

        urange = np.arange(origin[0], limit[0], step[0])
        vrange = np.arange(origin[1], limit[1], step[1])

        usize = urange.size
        vsize = vrange.size

        #------------------------------------------
        # Construct the empty array of values
        #------------------------------------------
        shape_list = len(self.shape) * [1]
        if self.u_axis >= 0:
            shape_list[self.u_axis] = usize
        if self.v_axis >= 0:
            shape_list[self.v_axis] = vsize

        values = np.empty(tuple(shape_list + [2]))

        #-----------------------
        # Populate the array
        #-----------------------
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

        #-------------------------
        # Return the Meshgrid
        #-------------------------
        grid = Pair(values)
        return Meshgrid(self.fov, grid, fov_keywords)
    #===========================================================================



    #===========================================================================
    # timegrid
    #===========================================================================
    def timegrid(self, meshgrid, oversample=1, tfrac=(0,1)):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a Scalar of times broadcastable with the shape of the given
        meshgrid.

        Input:
            meshgrid    the meshgrid defining spatial sampling.
            oversample  1 to obtain one time sample per pixel; > 1 for finer
                        sampling in time.

            tfrac       a tuple interpreted in different ways depending on the
                        observation's structure.
                        - if this observation has no time-dependence, it is the
                          pair of fractional time limits within the overall
                          exposure duration.
                        - if this observation has time-dependence that is
                          entirely coupled to spatial axes, then it is the the
                          fractional time limits within each pixel's individual
                          exposure duration.
                        - if this observation has time-dependence that is
                          entirely decoupled from the spatial axes, then it is
                          the start and end time relative to the time limits of
                          the defined cadence.
                        - the possible case of a 2-D time-dependence that has
                          only one axis coupled to a spatial axis is not
                          supported.
        """
# ???                        This argument is overridden by a tfrac property.
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ???
#         if hasattr(self, 'tfrac'): tfrac = self.tfrac
#         else :
#             if isinstance(tfrac, numbers.Number):
#                 tfrac = (tfrac, tfrac)
        if isinstance(tfrac, numbers.Number):
            tfrac = (tfrac, tfrac)

        #-------------------------------------------
        # Handle a time-independent observation
        #-------------------------------------------
        if self.t_axis == -1:

            time0 = self.time[0] + tfrac[0] * (self.time[1] - self.time[0])
            time1 = self.time[0] + tfrac[1] * (self.time[1] - self.time[0])

            # One step implies midtime, which can be returned as a scalar
            if oversample == 1:
                return Scalar(0.5 * (time0 + time1))

            # Otherwise, uniform time steps between endpoints
            fracs = np.arange(oversample) / (oversample - 1.)
            times = time0 + fracs * (time1 - time0)

            # Time is on a leading axis
            tshape = times.shape + len(self.shape) * (1,)
            return Scalar.as_scalar(times.reshape(tshape))

        #----------------------------------------
        # Get times at each pixel in meshgrid
        #----------------------------------------
        (tstarts, tstops) = self.times_at_uv(meshgrid.uv)

        #-------------------------
        # Scale based on tfrac
        #-------------------------
        time0 = tstarts + tfrac[0] * (tstops - tstarts)
        time1 = tstarts + tfrac[1] * (tstops - tstarts)

        #-----------------------
        # Handle 1-D case
        #-----------------------
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
            tstep0 = tfrac[0] * self.cadence.shape[0]
            tstep1 = tfrac[1] * self.cadence.shape[0]
            tsteps = np.arange(tstep0, tstep1 + 1.e-10, 1./oversample)
            times = self.cadence.time_at_tstep(tsteps)

            shape_list = len(self.shape) * [1]
            shape_list[self.t_axis] = len(times)
            times = Scalar.as_scalar(times).reshape(tuple(shape_list))
            return times

        #------------------------------
        # Handle a 2-D observation
        #------------------------------
        if (self.t_axis[0] not in (self.u_axis, self.v_axis) or
            self.t_axis[1] not in (self.u_axis, self.v_axis)):
                raise NotImplementedError('timegrid not implemented for ' +
                                          't axes (%d,%d), ' % self.t_axis,
                                          'u axis %d, ' % self.u_axis,
                                          'v axis %d'   % self.v_axis)

        # Time aligns with u-axis AND v-axis

        #---------------------------------
        # One time step implies midtime
        #---------------------------------
        if oversample == 1:
            return Scalar.as_scalar(0.5 * (time0 + time1))

        #---------------------------------------------------
        # Otherwise, uniform time steps on a leading axis
        #---------------------------------------------------
        fracs = np.arange(oversample) / (oversample - 1.)
        fracs = fracs.reshape(fracs.shape + len(self.shape) * (1,))
        return Scalar(time0 + fracs * (time1 - time0))
    #===========================================================================



    #===========================================================================
    # event_at_grid
    #===========================================================================
    def event_at_grid(self, meshgrid=None, time=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a photon arrival event from directions defined by a meshgrid.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view; None for a directionless observation.
            time        a Scalar of times; None to use the midtime of each
                        pixel in the meshgrid.

        Return:         the corresponding event.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if time is None:
            time = self.midtime_at_uv(meshgrid.uv)

        time = Scalar.as_scalar(time)
        event = Event(time, Vector3.ZERO, self.path, self.frame)

        #--------------------------------------
        # Insert the arrival directions
        #--------------------------------------
        event.neg_arr_ap = meshgrid.los

        return event
    #===========================================================================



    #===========================================================================
    # gridless_event
    #===========================================================================
    def gridless_event(self, meshgrid=None, time=None, shapeless=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a photon arrival event irrespective of the direction.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view; None for a directionless observation. Here, it
                        is only used to define the times when time is None.
            time        a Scalar of times; None to use the midtime of each
                        pixel in the meshgrid.
            shapeless   True to return a shapeless event, referring to the mean
                        of all the times.

        Return:         the corresponding event.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if time is None:
            time = self.midtime_at_uv(meshgrid.uv)

        if shapeless:
            time = time.mean()

        return Event(time, Vector3.ZERO, self.path, self.frame)
    #===========================================================================



    #===========================================================================
    # uv_from_ra_and_dec
    #===========================================================================
    def uv_from_ra_and_dec(self, ra, dec, derivs=False, iters=2, quick={},
                           apparent=True, time_frac=0.5):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Convert arbitrary scalars of RA and dec to FOV (u,v) coordinates.

        Input:
            ra          a Scalar of J2000 right ascensions.
            dec         a Scalar of J2000 declinations.
            derivs      True to propagate derivatives of ra and dec through to
                        derivatives of the returned (u,v) Pairs.
            iters       the number of iterations to perform until convergence
                        is reached. Two is probably the most that should ever be
                        needed; Snapshot can override to one.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            apparent    True to interpret the (RA,dec) values as apparent
                        coordinates; False to interpret them as actual
                        coordinates. Default is True.
            time_frac   fractional time from the beginning to the end of the
                        time spent inside the selected pixel. 0. for the
                        beginning; 0.5 for the midtime, 1. for the end time.

        Return:         a Pair of (u,v) coordinates.

        Note: The only reasons for iteration are that the C-matrix and the
        velocity WRT the SSB could vary during the observation. I doubt this
        would ever be significant.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #---------------------------------------------------
        # Convert to line of sight in SSB/J2000 frame
        #---------------------------------------------------
        neg_arr_j2000 = Vector3.from_ra_dec_length(ra, dec, recursive=derivs)

        #--------------------------------------------------------------
        # Require extra at least two iterations if time_frac != 0.5
        #--------------------------------------------------------------
        if time_frac != 0.5:
            iters = max(2, iters)

        #----------------------------------------
        # Iterate until (u,v) has converged
        #---------------------------------------
        obs_time = self.midtime     # starting guess
        uv = None
        for iter in range(iters):

            # Define the photon arrival event
            obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)

            if apparent:
                obs_event.neg_arr_ap_j2000 = neg_arr_j2000
            else:
                obs_event.neg_arr_j2000 = neg_arr_j2000

            # Convert to FOV coordinates
            prev_uv = uv
            uv = self.fov.uv_from_los(obs_event.neg_arr_ap)

            # Update the time
            (t0,t1) = self.times_at_uv(uv)
            obs_time = t0 + time_frac * (t1 - t0)

            # Stop at convergence
            if uv == prev_uv: break

        return uv
    #===========================================================================



    #===========================================================================
    # uv_from_path
    #===========================================================================
    def uv_from_path(self, path, derivs=False, quick={}, converge={},
                           time_frac=0.5):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the (u,v) indices of an object in the FOV, given its path.

        Note: This procedure assumes that movement along a path is very limited
        during the exposure time of an individual pixel. It could fail to
        converge if there is a large gap in timing between adjacent pixels at a
        time when the object is crossing that gap. However, even then, it should
        select roughly the correct location. It could also fail to converge
        during a fast slew.

        Input:
            path        a Path object.
            derivs      True to propagate derivatives of the link time and
                        position into the returned event.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.
            time_frac   fractional time from the beginning to the end of the
                        time spent inside the selected pixel. 0. for the
                        beginning; 0.5 for the midtime, 1. for the end time.

        Return:
            uv_pair     the (u,v) indices of the pixel in which the point was
                        found. The path is evaluated at the mid-time of this
                        pixel.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-------------------------------------
        # Assemble convergence parameters
        #-------------------------------------
        if converge:
            defaults = PATH_PHOTONS.__dict__.copy()
            defaults.update(converge)
            converge = defaults
        else:
            converge = PATH_PHOTONS.__dict__

        iters = converge['max_iterations']
        precision = converge['dlt_precision']
        limit = converge['dlt_limit']

        #------------------------------------------------------------------
        # Require extra at least two iterations if time_frac != 0.5
        #------------------------------------------------------------------
        if time_frac != 0.5:
            iters = max(2, iters)

        #-----------------------------
        # Iterate to solution...
        #-----------------------------
        guess = None
        max_dt = np.inf
        obs_time = self.midtime     # starting guess

        for iter in range(iters):

            # Locate the object in the field of view
            obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)
            (path_event, obs_event) = path.photon_to_event(obs_event,
                                        derivs=False, guess=guess,
                                        quick=quick, converge=converge)
            guess = path_event.time
            (uv_min, uv_max) = self.uv_at_time(obs_event.time)

            # Update the observation times based on pixel midtimes
            (t0, t1) = self.times_at_uv(uv_min)
            new_obs_time = t0 + time_frac * (t1 - t0)

            # Test for convergence
            prev_max_dt = max_dt
            max_dt = abs(new_obs_time - obs_time).max()
            obs_time = new_obs_time

            if LOGGING.observation_iterations:
                print(LOGGING.prefix, "Observation.uv_from_path", iter, max_dt)

            if max_dt <= PATH_PHOTONS.dlt_precision or max_dt >= prev_max_dt:
                break

        #------------------------------------------
        # Return the results at the best mid-time
        #-------------------------------------------
        obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)

        (path_event, obs_event) = path.photon_to_event(obs_event,
                                        derivs=derivs, guess=guess,
                                        quick=quick, converge=converge)

        return self.fov.uv_from_los(obs_event.neg_arr_ap, derivs=derivs)
    #===========================================================================



    #===========================================================================
    # inventory
    #===========================================================================
    def inventory(self, bodies, expand=0., return_type='list', fov=None,
                        quick={}, converge={}, time_frac=0.5):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the body names that appear unobscured inside the FOV.

        Restrictions: All inventory calculations are performed at a single
        observation time specified by time_frac. All bodies are assumed to be
        spherical.

        Input:
            bodies      a list of the names of the body objects to be included
                        in the inventory.
            expand      an optional angle in radians by which to extend the
                        limits of the field of view. This can be used to
                        accommodate pointing uncertainties.
            return_type 'list' returns the inventory as a list of names.
                        'flags' returns the inventory as an array of boolean
                                flag values in the same order as bodies.
                        'full' returns the inventory as a dictionary of
                                dictionaries. The main dictionary is indexed by
                                body name. The subdictionaries contain
                                attributes of the body in the FOV; see below.
            fov         use this fov; if None, use self.fov.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            converge    an optional dictionary of parameters to override the
                        configured default convergence parameters. The default
                        configuration is defined in config.py.
            time_frac   fractional time from the beginning to the end of the
                        observation for which the inventory applies. 0. for the
                        beginning; 0.5 for the midtime, 1. for the end time.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        raise NotImplementedError(
                'Observation subclass "%s" ' % type(self).__name__ +
                'does not implement method inventory()')
    #===========================================================================


#*******************************************************************************



################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_Observation
#*******************************************************************************
class Test_Observation(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        # TBD
        # Note in particular that uv_from_path() is incomplete and untested!

        pass
    #===========================================================================


#*******************************************************************************

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
