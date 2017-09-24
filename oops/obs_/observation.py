################################################################################
# oops/obs_/observation.py: Abstract class Observation
################################################################################

import numpy as np
import numbers
from polymath import *

from oops.config          import LOGGING, PATH_PHOTONS
from oops.event           import Event
from oops.frame_.frame    import Frame
from oops.meshgrid        import Meshgrid
from oops.path_.multipath import MultiPath
from oops.body            import Body

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
        uv_shape        a list or tuple defining the 2-D shape of the data array
                        in (u,v) order. Note that this may differ from
                        fov.uv_shape.
        u_axis, v_axis  integers identifying the axes of the data array
                        associated with the u-axis and the v-axis. Use -1 if
                        that axis is not associated with an array index.
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

    ####################################################
    # Methods to be defined for each subclass
    ####################################################

    def __init__(self):
        """A constructor."""

        pass

    def uvt(self, indices, fovmask=False):
        """Return coordinates (u,v) and time t for indices into the data array.

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

        raise NotImplementedException("uvt() is not implemented")

    def uvt_range(self, indices, fovmask=False):
        """Return ranges of coordinates and time for integer array indices.

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

        raise NotImplementedException("uvt_range() is not implemented")

    def indices_at_uvt(self, uv_pair, time, fovmask=False):
        """Return a vector of indices for given FOV coordinates (u,v) and time.

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

        raise NotImplementedException("indices_at_uvt() is not implemented")

    def times_at_uv(self, uv_pair, fovmask=False):
        """Return start and stop times of the specified spatial pixel (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        raise NotImplementedException("times_at_uv() is not implemented")

    def uv_at_time(self, time, fovmask=False):
        """The (u,v) range of spatial pixels observed at the specified time.

        Input:
            time        a Scalar of time values in seconds TDB.
            tmask       True to mask values outside the time limits.

        Return:         (uv_min, uv_max)
            uv_min      the lower (u,v) corner of the area observed at the
                        specified time.
            uv_max      the upper (u,v) corner of the area observed at the
                        specified time.
        """

        raise NotImplementedException("uv_at_time() is not implemented")

    def sweep_duv_dt(self, uv_pair):
        """Return the mean local sweep speed of the instrument along (u,v) axes.

        Input:
            uv_pair     a Pair of spatial indices (u,v).

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        raise NotImplementedException("sweep_duv_dt() is not implemented")

    def time_shift(self, dtime):
        """Return a copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        raise NotImplementedException("time_shift() is not implemented")

    ####################################################
    # Subfield support methods
    ####################################################

    def insert_subfield(self, key, value):
        """Adds a given subfield to the Event."""

        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

    def delete_subfield(self, key):
        """Deletes a subfield, but not arr or dep."""

        if key in self.subfields:
            del self.subfields[key]
            del self.__dict__[key]

    def delete_subfields(self):
        """Deletes all subfields."""

        for key in self.subfields:
            del self.subfields[key]
            del self.__dict__[key]

    ####################################################
    # Methods probably not requiring overrides
    ####################################################

    def uv_is_outside(self, uv_pair, inclusive=True):
        """Return a boolean mask identifying coordinates outside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.

        Return:         a boolean NumPy array indicating True where the point is
                        outside the FOV.
        """

        return self.fov.uv_is_outside(uv_pair, inclusive)

    def midtime_at_uv(self, uv):
        """Return the mid-time for the selected spatial pixel (u,v).

        Input:
            uv          a Pair of (u,v) coordinates.
        """

        (time0, time1) = self.times_at_uv(uv)
        return 0.5 * (time0 + time1)

    def meshgrid(self, origin=0.5, undersample=1, oversample=1, limit=None,
                       fov_keywords={}):
        """Return a Meshgrid shaped to broadcast to the observation's shape.

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
        if limit is None: limit = self.fov.uv_shape
        if isinstance(limit, numbers.Number): limit = (limit,limit)
        limit = Pair.as_pair(limit).values.astype('float')

        if isinstance(origin, numbers.Number): origin = (origin, origin)
        origin = Pair.as_pair(origin).values.astype('float')

        if isinstance(undersample, numbers.Number):
            undersample = (undersample, undersample)
        undersample = Pair.as_pair(undersample).values.astype('float')

        if isinstance(oversample, numbers.Number):
            oversample = (oversample, oversample)
        oversample = Pair.as_pair(oversample).values.astype('float')

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

    def timegrid(self, origin=None, undersample=1, oversample=1, limit=None):
        """Return a Scalar of times broadcastable to the shape of the
        observation.

        If an observation has no time-dependence associated with an axis and
        only one time is to be returned, the returned value is a float, not a
        Scalar.

        If an observation has no time-dependence associated with an axis but
        multiple times are requested, the shape of the returned Scalar has a
        leading axis to encompass the time dependence.

        Input:
            origin      A single value or tuple (for N-dimensional cadences)
                        defining the origin of the time grid, in units of the
                        index. Default is None, which centers the time samples
                        around the observation midtime.

            undersample A single value or tuple defining the magnitude of
                        under-sampling to be performed. For example, a value of
                        2 would cause the timegrid to sample every other time
                        step along each axis.

            oversample  A single value or tuple defining the magnitude of
                        over-sampling to be performed. For example, a value of
                        2 would create a two time steps along each axis of each
                        time step.

            limit       A single value or tuple defining the upper limits of the
                        timegrid. By default, this is the shape of the cadence.

        """

        # Validate input parameters...

        # 0-D or 1-D case: convert to scalars
        if isinstance(self.t_axis, numbers.Number):
            if not isinstance(undersample, numbers.Number):
                assert len(undersample) == 1
                undersample = undersample[0]

            if not isinstance(oversample, numbers.Number):
                assert len(oversample) == 1
                oversample = oversample[0]

            undersample = float(undersample)
            oversample = float(oversample)
            step = undersample / oversample

            if not isinstance(limit, numbers.Number):
                if limit is None:
                    limit = self.cadence.shape[0]
                else:
                    assert len(limit) == 1
                    limit = limit[0]

            limit = float(limit)

            if not isinstance(origin, numbers.Number):
                if origin is None:
                    origin = 0.5 * limit * step
                else:
                    assert len(origin) == 1
                    origin = origin[0]

            origin = float(origin)

        # 2-D case: convert to arrays
        else:
            axes = len(self.t_axis)

            if isinstance(undersample, numbers.Number):
                undersample = axes * [undersample,]

            if isinstance(oversample, numbers.Number):
                oversample = axes * [oversample,]

            undersample = np.asfarray(undersample)
            oversample = np.asfarray(oversample)
            step = undersample / oversample

            if limit is None:
                limit = self.cadence.shape
            elif isinstance(limit, numbers.Number):
                limit = axes * [limit,]

            limit = np.asfarray(limit)

            if isinstance(origin, numbers.Number):
                if origin is None:
                    origing = 0.5 * limit * step
                else:
                    origin = axes * [origin,]

            origin = np.asfarray(origin)

        # Handle observations without time dependence
        if self.t_axis == -1:
            limit += step * 1.e-10      # Allow a little slop at the top

            # Tabulate the times
            tstep = np.arange(origin, limit, step)
            times = self.time[0] * (1. - tstep) + self.time[1] * tstep

            # For no times, return observation midtime
            if len(times) == 0:
                return self.midtime

            # For one time, return a float
            if len(times) == 1:
                return times[0]

            # Otherwise, return the reshaped Scalar
            times = times.reshape(times.shape + len(self.shape) * (1,))
            return Scalar(times)

        # Handle 1-D case
        if isinstance(self.t_axis, numbers.Number):
            limit += step * 1.e-10   # Allow a little slop at the top

            # Tabulate the times
            tstep = Scalar(np.arange(origin, limit, step))
            times = self.cadence.time_at_tstep(tstep, mask=False)

            # Re-shape
            shape_list = len(self.shape) * [1]
            shape_list[self.t_axis] = times.shape[0]
            values = times.values.reshape(tuple(shape_list))

            # Return the Scalar
            return Scalar(values)

        # Handle N-dimensional case...

        # Construct the index arrays
        limit += step * 1.e-10      # Allow a little slop at the upper end

        ranges = []
        sizes = []
        shape_list = axes * [1]
        for i in range(axes):
            ranges.append(np.arange(origin[i], limit[i], step[i]))
            sizes.append(len(ranges[i]))
            shape_list[self.t_axis[i]] = sizes[i]

        # Construct the empty array of values in the desired shape
        values = np.empty(tuple(shape_list) + (axes,))

        # Populate the tstep array
        for i in range(axes):
            temp_shape = axes * [1]
            temp_shape[self.t_axis[i]] = sizes[i]
            tsteps = ranges[i].reshape(tuple(temp_shape))
            values[...,i] = tsteps

        # Return the times
        tsteps = Vector(values)
        return self.cadence.time_at_tstep(Vector(values), mask=False)

    def event_at_grid(self, meshgrid=None, time=None):
        """Return a photon arrival event from directions defined by a meshgrid.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view; None for a directionless observation.
            time        a Scalar of times; None to use the midtime of each
                        pixel in the meshgrid.

        Return:         the corresponding event.
        """

        if time is None:
            time = self.midtime_at_uv(meshgrid.uv)

        time = Scalar.as_scalar(time)
        event = Event(time, Vector3.ZERO, self.path, self.frame)

        # Insert the arrival directions
        event.neg_arr_ap = meshgrid.los

        return event

    def gridless_event(self, meshgrid=None, time=None, shapeless=False):
        """Return a photon arrival event irrespective of the direction.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view; None for a directionless observation.
            time        a Scalar of times; None to use the midtime of each
                        pixel in the meshgrid.
            shapeless   True to return a shapeless event, referring to the mean
                        of all the times.

        Return:         the corresponding event.
        """

        if time is None:
            time = self.midtime_at_uv(meshgrid.uv)

        if shapeless:
            time = time.mean()

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        return event

    def uv_from_ra_and_dec(self, ra, dec, derivs=False, iters=2, quick={},
                           apparent=True, time_frac=0.5):
        """Convert arbitrary scalars of RA and dec to FOV (u,v) coordinates.

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

        # Convert to line of sight in SSB/J2000 frame
        neg_arr_j2000 = Vector3.from_ra_dec_length(ra, dec, recursive=derivs)

        # Require extra at least two iterations if time_frac != 0.5
        if time_frac != 0.5:
            iters = max(2, iters)

        # Iterate until (u,v) has converged
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

    def uv_from_path(self, path, derivs=False, quick={}, converge={},
                           time_frac=0.5):
        """Return the (u,v) indices of an object in the FOV, given its path.

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

        # Assemble convergence parameters
        if converge:
            defaults = PATH_PHOTONS.__dict__.copy()
            defaults.update(converge)
            converge = defaults
        else:
            converge = PATH_PHOTONS.__dict__

        iters = converge['max_iterations']
        precision = converge['dlt_precision']
        limit = converge['dlt_limit']

        # Require extra at least two iterations if time_frac != 0.5
        if time_frac != 0.5:
            iters = max(2, iters)

        # Iterate to solution...
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
                print LOGGING.prefix, "Observation.uv_from_path", iter, max_dt

            if max_dt <= PATH_PHOTONS.dlt_precision or max_dt >= prev_max_dt:
                break

        # Return the results at the best mid-time
        obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)

        (path_event, obs_event) = path.photon_to_event(obs_event,
                                        derivs=derivs, guess=guess,
                                        quick=quick, converge=converge)

        return self.fov.uv_from_los(obs_event.neg_arr_ap, derivs=derivs)

    ### NOTE: This general version of uv_from_path() has not been tested!
    ### This method will at least need an override for the Pixel class.

    def inventory(self, bodies, expand=0., return_type='list', fov=None,
                        quick={}, converge={}, time_frac=0.5):
        """Return the body names that appear unobscured inside the FOV.

        Restrictions: All inventory calculations are performed at a single
        observation time specified by time_frac. All bodies are assumed to be
        spherical.

        Input:
            bodies      a list of the names of the body objects to be included
                        in the inventory.
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

        assert return_type in ('list', 'flags', 'full')

        if fov is None:
            fov = self.fov

        body_names = [Body.as_body_name(body) for body in bodies]
        bodies  = [Body.as_body(body) for body in bodies]
        nbodies = len(bodies)

        path_ids = [body.path for body in bodies]
        multipath = MultiPath(path_ids)

        obs_time = self.time[0] + time_frac * (self.time[1] - self.time[0])
        obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)
        (_,
         arrival_event) = multipath.photon_to_event(obs_event, quick=quick,
                                                    converge=converge)

        centers = arrival_event.neg_arr_ap
        ranges = centers.norm()
        radii = Scalar([body.radius for body in bodies])
        radius_angles = (radii/ranges).arcsin()

        inner_radii = Scalar([body.inner_radius for body in bodies])
        inner_angles = (inner_radii / ranges).arcsin()

        # This array equals True for each body falling somewhere inside the FOV
        falls_inside = np.empty(nbodies, dtype='bool')
        for i in range(nbodies):
            falls_inside[i] = fov.sphere_falls_inside(centers[i], radii[i])

        # This array equals True for each body completely hidden by another
        is_hidden = np.zeros(nbodies, dtype='bool')
        for i in range(nbodies):
          if not falls_inside[i]: continue

          for j in range(nbodies):
            if not falls_inside[j]: continue

            if ranges[i] < ranges[j]: continue
            if radius_angles[i] > inner_angles[j]: continue

            sep = centers[i].sep(centers[j])
            if sep < inner_angles[j] - radius_angles[i]:
                is_hidden[i] = True

        flags = falls_inside & ~is_hidden

        # Return as flags
        if return_type == 'flags':
            return flags

        # Return as list
        if return_type == 'list':
            ret_list = []
            for i in range(nbodies):
                if flags[i]: ret_list.append(body_names[i])
            return ret_list

        # Return full info
        ret_dict = {}

        u_scale = fov.uv_scale.vals[0]
        v_scale = fov.uv_scale.vals[1]
        body_uv = fov.uv_from_los(arrival_event.neg_arr_ap).vals
        for i in range(nbodies):
            if flags[i]:
                body_data = {}
                body_data['name'] = body_names[i]
                body_data['center_uv'] = body_uv[i]
                body_data['center'] = centers[i].vals
                body_data['range'] = ranges[i].vals
                body_data['outer_radius'] = radii[i].vals
                body_data['inner_radius'] = inner_radii[i].vals
                u_res = ranges[i] * self.fov.uv_scale.to_scalar(0).tan()
                v_res = ranges[i] * self.fov.uv_scale.to_scalar(1).tan()
                body_data['resolution'] = Pair.from_scalars(u_res, v_res).vals
                u = body_uv[i][0]
                v = body_uv[i][1]
                body_data['u_min_unclipped'] = np.floor(
                                    u-radius_angles[i].vals/u_scale)
                body_data['u_max_unclipped'] = np.ceil(
                                    u+radius_angles[i].vals/u_scale)
                body_data['v_min_unclipped'] = np.floor(
                                    v-radius_angles[i].vals/v_scale)
                body_data['v_max_unclipped'] = np.ceil(
                                    v+radius_angles[i].vals/v_scale)
                body_data['u_min'] = np.clip(body_data['u_min_unclipped'],
                                             0, self.data.shape[1]-1)
                body_data['u_max'] = np.clip(body_data['u_max_unclipped'],
                                             0, self.data.shape[1]-1)
                body_data['v_min'] = np.clip(body_data['v_min_unclipped'],
                                             0, self.data.shape[0]-1)
                body_data['v_max'] = np.clip(body_data['v_max_unclipped'],
                                             0, self.data.shape[0]-1)
                body_data['u_pixel_size'] = radius_angles[i].vals/u_scale*2
                body_data['v_pixel_size'] = radius_angles[i].vals/v_scale*2
                
                # Final sanity check - the moon HAS to be actually inside the
                # FOV. There are times previous tests fail when we are really
                # close to the moon. (See Enceladus in N1669812089_1 for
                # an example)
                if (body_data['u_min_unclipped'] >= self.data.shape[1] or
                    body_data['u_max_unclipped'] < 0 or
                    body_data['v_min_unclipped'] >= self.data.shape[0] or
                    body_data['v_max_unclipped'] < 0):
                    continue
                ret_dict[body_names[i]] = body_data

        return ret_dict

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
