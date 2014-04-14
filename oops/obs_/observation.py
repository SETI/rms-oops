################################################################################
# oops/obs_/observation.py: Abstract class Observation
################################################################################

import numpy as np
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

    def uvt(self, indices, fovmask=False, **keywords):
        """Return coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Tuple of array indices.
            fovmask     True to mask values outside the field of view.
            keywords    optional additional keyword values required to define
                        the FOV and the relationship between (u,v) and time.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        raise NotImplementedException("uvt() is not implemented")

    def uvt_range(self, indices, fovmask=False, **keywords):
        """Return ranges of coordinates and time for integer array indices.

        Input:
            indices     a Tuple of integer array indices.
            fovmask     True to mask values outside the field of view.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:         (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of (u,v) associated
                        the pixel.
            uv_max      a Pair defining the maximum values of (u,v).
            time_min    a Scalar defining the minimum time associated with the
                        pixel. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        raise NotImplementedException("uvt_range() is not implemented")

    def indices_at_uvt(self, uv_pair, time, fovmask=False, **keywords):
        """Return a vector of indices for given FOV coordinates (u,v) and time.

        This method supports non-integer positions and time steps, and returns
        fractional indices.

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in or near the field
                        of view.
            time        a Scalar of times in seconds TDB.
            fovmask     True to mask values outside the field of view.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:
            indices     a Tuple of array indices. Any array indices not
                        constrained by (u,v) or time are returned with value 0.
                        Note that returned indices can fall outside the nominal
                        limits of the data object.
        """

        raise NotImplementedException("indices_at_uvt() is not implemented")

    def times_at_uv(self, uv_pair, fovmask=False, **keywords):
        """Return start and stop times of the specified spatial pixel (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        raise NotImplementedException("times_at_uv() is not implemented")

    def uv_at_time(self, time, fovmask=False, **keywords):
        """Return the active range of FOV (u,v) coordinates at a given time.

        Input:
            uv_pair     a Scalar of time values in seconds TDB.
            fovmask     True to mask values outside the time limits and/or the
                        field of view.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:         (uv_min, uv_max)
            uv_min      the lower (u,v) corner of the area observed at the
                        specified time.
            uv_max      the upper (u,v) corner of the area observed at the
                        specified time.
        """

        raise NotImplementedException("uv_at_time() is not implemented")

    def sweep_duv_dt(self, uv_pair, **keywords):
        """Return the mean local sweep speed of the instrument along (u,v) axes.

        Input:
            uv_pair     a Pair of spatial indices (u,v).
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

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

        if key in ('arr','dep'):
            self.subfields[key] = Empty()
            self.__dict__[key] = self.subfields[key]
        elif self.subfields.has_key(key):
            del self.subfields[key]
            del self.__dict__[key]

    def delete_subfields(self):
        """Deletes all subfields."""

        for key in self.subfields.keys():
            if key not in ('arr','dep'):
                del self.subfields[key]
                del self.__dict__[key]

    ####################################################
    # Methods probably not requiring overrides
    ####################################################

    def uv_is_outside(self, uv_pair, inclusive=True, **keywords):
        """Return a boolean mask identifying coordinates outside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:         a boolean NumPy array indicating True where the point is
                        outside the FOV.
        """

        return self.fov.uv_is_outside(uv_pair, inclusive, **keywords)

    def midtime_at_uv(self, uv, **keywords):
        """Return the mid-time for the selected spatial pixel (u,v).

        Input:
            uv          a Pair of (u,v) coordinates.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Optional additional parameters affecting the timing and FOV can be
        passed as keywords."""

        (time0, time1) = self.times_at_uv(uv, **keywords)
        return 0.5 * (time0 + time1)

    def event_at_grid(self, meshgrid=None, time=None, **keywords):
        """Return a photon arrival event from directions defined by a meshgrid.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view; None for a directionless observation.
            time        a Scalar of times; None to use the midtime of each
                        pixel in the meshgrid.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:         the corresponding event.
        """

        if time is None:
            time = self.midtime_at_uv(meshgrid.uv, **keywords)

        event = Event(time, (Vector3.ZERO,Vector3.ZERO), self.path, self.frame)

        # Insert the arrival directions
        event.insert_subfield('arr', -meshgrid.los)

        return event

    def gridless_event(self, meshgrid=None, time=None, shapeless=False,
                             **keywords):
        """Return a photon arrival event irrespective of the direction.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view; None for a directionless observation.
            time        a Scalar of times; None to use the midtime of each
                        pixel in the meshgrid.
            shapeless   True to return a shapeless event, referring to the mean
                        of all the times.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:         the corresponding event.
        """

        if time is None:
            time = self.midtime_at_uv(meshgrid.uv, **keywords)

        if shapeless:
            time = time.mean()

        event = Event(time, (Vector3.ZERO,Vector3.ZERO), self.path, self.frame)

        return event

    def uv_from_ra_and_dec(self, ra, dec, derivs=False, iters=2, quick={},
                                 **keywords):
        """Convert arbitrary scalars of RA and dec to FOV (u,v) coordinates.

        Input:
            ra          a Scalar of J2000 right ascensions.
            dec         a Scalar of J2000 declinations.
            derivs      True to propagate derivatives of ra and dec through to
                        derivatives of the returned (u,v) Pairs.
            iters       the number of iterations to perform until convergence
                        is reached. Two is the most that should ever be needed;
                        Snapshot should override to one.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:         a Pair of (u,v) coordinates.

        Note: The only reasons for iteration are that the C-matrix and the
        velocity WRT the SSB could vary during the observation. I doubt this
        would ever be significant.
        """

        # Convert to scalar; strip derivatives if necessary
        ra = Scalar.as_scalar(ra, derivs)
        dec = Scalar.as_scalar(dec, derivs)

        # Convert to line of sight in SSB/J2000 frame
        cos_dec = dec.cos()
        x = cos_dec * ra.cos()
        y = cos_dec * ra.sin()
        z = dec.sin()
        los = -Vector3.from_scalars(x,y,z)      # negative for incoming photon

        # Define the rotation from J2000 to the observer's frame
        rotation = self.frame.wrt(Frame.J2000)

        # Iterate until the observation time has converged
        obs_time = self.midtime
        for iter in range(iters):

            # Define the photon arrival event
            obs_event = Event(obs_time, (Vector3.ZERO, Vector3.ZERO),
                              self.path, self.frame)

            # Use a quickframe if appropriate
            rotation = Frame.quick_frame(rotation, obs_time, quick)

            # Rotate the line of sight to the observer's frame
            xform = rotation.transform_at_time(obs_time)
            obs_event.insert_subfield("arr", xform.rotate(los))

            # Get the apparent direction, allowing for stellar aberration
            apparent_arr = obs_event.apparent_arr(quick=quick)

            # Convert to FOV coordinates
            uv = self.fov.uv_from_los(apparent_arr)

            # Update the time
            obs_time = self.midtime_at_uv(uv, **keywords)

        return uv

    # This procedure assumes that movement along a path is very limited during
    # the exposure time of an individual pixel. It could fail to converge if
    # there is a large gap in timing between adjacent pixels at a time when the
    # object is crossing that gap. However, even then, it should select roughly
    # the correct location. It could also fail to converge during a fast slew.

    ##### DOES NOT WORK, IN PROGRESS!

    def uv_from_path(self, path, derivs=False, quick=False, converge={},
                           **keywords):
        """Return the (u,v) indices of an object in the FOV, given its path.

        Input:
            path        a Path object.
            extras      a tuple of Scalar index values defining any extra
                        indices into the observation's array, should these
                        values be relevant.
            quick       defines how to use QuickPaths and QuickFrames.
            derivs      True to include derivatives d(u,v)/dt, neglecting any
                        sweep motion within the observation.
            keywords    optional additional keyword values, based on the
                        requirements of the observation and its attributes.

        Return:
            uv_pair     the (u,v) indices of the pixel in which the point was
                        found. The path is evaluated at the mid-time of this
                        pixel.
        """

        if iters is None: iters = PATH_PHOTONS.max_iterations

        guess = None
        max_dt = np.inf
        obs_time = self.midtime

        # Iterate until convergence or until the limit is reached...
        for iter in range(iters):

            # Locate the object in the field of view
            obs_event = Event(obs_time, (Vector3.ZERO,Vector3.ZERO),
                              self.path, self.frame)
            (path_event,obs_event) = path.photon_to_event(obs_event,
                                        derivs=False, guess=guess,
                                        quick=quick, converge=converge)
            guess = path_event.time
            uv = self.uv_from_event(obs_event)      # not implemented??

            # Update the times based on the locations
            new_obs_time = self.midtime_at_uv(uv)

            # Test for convergence
            prev_max_dt = max_dt
            max_dt = abs(new_obs_time - obs_time).max()
            obs_time = new_obs_time

            if LOGGING.observation_iterations:
                print LOGGING.prefix, "Observation.uv_from_path", iter, max_dt

            if max_dt <= PATH_PHOTONS.dlt_precision or max_dt >= prev_max_dt:
                break

        # Return the results at the best mid-time
        obs_event = Event(obs_time, (Vector3.ZERO, Vector3.ZERO),
                          self.path, self.frame)

        (path_event,obs_event) = path.photon_to_event(obs_event,
                                        derivs=derivs, guess=guess,
                                        quick=quick, converge=converge)

        ### NEEDS FIXING BELOW

        uv = self.fov.uv_from_los(-obs_event.arr, derivs=derivs)
        # If derivs is True, then uv.d_dlos is defined

        # Combine the derivatives if necessary
        if derivs:
            duv_dt = obs_event.arr.d_dt/obs_event.arr.plain().norm() * uv.d_dlos
            uv.insert_subfield("d_dt", duv_dt)

        return uv

    def inventory(self, bodies, expand=0., as_list=False, as_flags=False):
        """Return the body names that appear unobscured inside the FOV.

        Input:
            bodies      a list of the names of the body objects to be included
                        in the inventory.
            expand      an optional angle in radians by which to extend the
                        limits of the field of view. This can be used to
                        accommodate pointing uncertainties.
            as_list     True to return the inventory as a list of names. This is
                        the default if neither as_list or as_flags is True.
            as_flags    True to return the inventory as an array of boolean flag
                        values.

        Return:         list, array, or (list,array)

            If as_list is True, it returns a list of the names of all the body
            objects that fall at least partially inside the FOV and are not
            completely obscured by another object in the list.

            If as_flags is True, it returns a boolean array containing True
            everywhere that the body falls at least partially inside the FOV
            and is not completely obscured.

            If both as_list and as_flags are True, then the tuple (list,array)
            is returned.

        Restrictions: All inventory calculations are performed at the
        observation midtime and all bodies are assumed to be spherical.
        """

        body_names = [Body.as_body_name(body) for body in bodies]
        bodies  = [Body.as_body(body) for body in bodies]
        nbodies = len(bodies)

        path_ids = [body.path_id for body in bodies]
        multipath = MultiPath(path_ids)

        obs_event = Event(self.midtime, (Vector3.ZERO,Vector3.ZERO),
                          self.path, self.frame)
        ignore = multipath.photon_to_event(obs_event)   # insert photon arrivals

        centers = -obs_event.arr
        ranges = centers.norm()
        radii = Scalar([body.radius for body in bodies])
        radius_angles = (radii / ranges).arcsin()

        inner_radii = Scalar([body.inner_radius for body in bodies])
        inner_angles = (inner_radii / ranges).arcsin()

        # This array equals True for each body falling somewhere inside the FOV
        falls_inside = np.empty(nbodies, dtype='bool')
        for i in range(nbodies):
            falls_inside[i] = self.fov.sphere_falls_inside(centers[i], radii[i])

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

        if as_flags and not as_list:
            return flags

        list = []
        for i in range(nbodies):
            if flags[i]: list.append(body_names[i])

        if not as_flags:
            return list
        else:
            return (list, flags)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Observation(unittest.TestCase):

    def runTest(self):

        # TBD
        # Note in particular that uv_from_path() is imcomplete and untested!

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
