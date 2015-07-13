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

#         if key in ('arr','dep'):
#             self.subfields[key] = Empty()
#             self.__dict__[key] = self.subfields[key]
#         elif self.subfields.has_key(key):
#             del self.subfields[key]
#             del self.__dict__[key]

        if key in self.subfields:
            del self.subfields[key]
            del self.__dict__[key]

    def delete_subfields(self):
        """Deletes all subfields."""

        for key in self.subfields:
#            if key not in ('arr','dep'):
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

        event = Event(time, (Vector3.ZERO,Vector3.ZERO), self.path, self.frame)

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

        event = Event(time, (Vector3.ZERO,Vector3.ZERO), self.path, self.frame)

        return event

    def uv_from_ra_and_dec(self, ra, dec, derivs=False, iters=2, quick={},
                           apparent=False):
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
            apparent    True to interpret the (RA,dec) values as apparent
                        coordinates; False to interpret them as actual
                        coordinates. Default is False.

        Return:         a Pair of (u,v) coordinates.

        Note: The only reasons for iteration are that the C-matrix and the
        velocity WRT the SSB could vary during the observation. I doubt this
        would ever be significant.
        """

        # Convert to line of sight in SSB/J2000 frame
        neg_arr_j2000 = Vector3.from_ra_dec_length(ra, dec, recursive=derivs)

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
            if apparent:
                obs_event.neg_arr_ap_j2000 = xform.rotate(neg_arr_j2000)
            else:
                obs_event.neg_arr_j2000 = xform.rotate(neg_arr_j2000)

            # Convert to FOV coordinates
            uv = self.fov.uv_from_los(obs_event.neg_arr_ap)

            # Update the time
            obs_time = self.midtime_at_uv(uv)

        return uv

    def uv_from_path(self, path, derivs=False, quick={}, converge={}):
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

        # Iterate to solution...
        guess = None
        max_dt = np.inf
        obs_time = self.midtime

        for iter in range(iters):

            # Locate the object in the field of view
            obs_event = Event(obs_time, (Vector3.ZERO,Vector3.ZERO),
                              self.path, self.frame)
            (path_event, obs_event) = path.photon_to_event(obs_event,
                                        derivs=False, guess=guess,
                                        quick=quick, converge=converge)
            guess = path_event.time
            (uv_min, uv_max) = self.uv_at_time(obs_event.time)

            # Update the observation times based on pixel midtimes
            new_obs_time = self.midtime_at_uv(uv_min)

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

        (path_event, obs_event) = path.photon_to_event(obs_event,
                                        derivs=derivs, guess=guess,
                                        quick=quick, converge=converge)

        return self.fov.uv_from_los(obs_event.neg_arr_ap, derivs=derivs)

    ### NOTE: This general version of uv_from_path() has not been tested!
    ### This method will at least need an override for the Pixel class.

    def inventory(self, bodies, expand=0., return_type='list', fov=None,
                        quick={}, converge={}):
        """Return the body names that appear unobscured inside the FOV.

        Restrictions: All inventory calculations are performed at the
        observation midtime and all bodies are assumed to be spherical.

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
                body_data['center_uv']     The U,V Pair of the center point
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
        """

        assert return_type in ('list', 'flags', 'full')

        if fov is None:
            fov = self.fov

        body_names = [Body.as_body_name(body) for body in bodies]
        bodies  = [Body.as_body(body) for body in bodies]
        nbodies = len(bodies)

        path_ids = [body.path for body in bodies]
        multipath = MultiPath(path_ids)

        obs_event = Event(self.midtime, (Vector3.ZERO,Vector3.ZERO),
                          self.path, self.frame)
        _, obs_event = multipath.photon_to_event(
                                    obs_event, quick=quick,
                                    converge=converge)  # insert photon arrivals

        centers = obs_event.neg_arr_ap
        ranges = centers.norm()
        radii = Scalar([body.radius for body in bodies])
        radius_angles = (radii / ranges).arcsin()

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
        body_uv = fov.uv_from_los(obs_event.neg_arr_ap).vals
        for i in range(nbodies):
            if flags[i]:
                body_data = {}
                body_data['name'] = body_names[i]
                body_data['center_uv'] = body_uv[i]
                body_data['center'] = centers[i]
                body_data['range'] = ranges[i]
                body_data['outer_radius'] = radii[i]
                body_data['inner_radius'] = inner_radii[i]
                u_res = ranges[i] * self.fov.uv_scale.to_scalar(0).tan()
                v_res = ranges[i] * self.fov.uv_scale.to_scalar(1).tan()
                body_data['resolution'] = Pair.from_scalars(u_res, v_res)
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
                ret_dict[body_names[i]] = body_data

        return ret_dict

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
