################################################################################
# oops/observation/snapshot.py: Subclass Snapshot of class Observation
################################################################################

import numpy as np
from polymath import Scalar, Pair, Vector, Vector3, Qube

from oops.observation       import Observation
from oops.body              import Body
from oops.cadence.metronome import Metronome
from oops.event             import Event
from oops.frame             import Frame
from oops.path              import Path
from oops.path.multipath    import MultiPath

class Snapshot(Observation):
    """A Snapshot is an Observation consisting of a 2-D image made up of pixels
    all exposed at the same time.
    """

    INVENTORY_IMPLEMENTED = True

    #===========================================================================
    def __init__(self, axes, tstart, texp, fov, path, frame, **subfields):
        """Constructor for a Snapshot.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'u' should
                        appear at the location of the array's u-axis; 'v' should
                        appear at the location of the array's v-axis. For
                        example, ('v','u'), is correct for a 2-D array read from
                        an image file in FITS or VICAR format.

            tstart      the start time of the observation in seconds TDB.

            texp        exposure duration of the observation in seconds.

            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y).

            path        the path waypoint co-located with the instrument.

            frame       the wayframe of a coordinate frame fixed to the optics
                        of the instrument. This frame should have its Z-axis
                        pointing outward near the center of the line of sight,
                        with the X-axis pointing rightward and the y-axis
                        pointing downward.

            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """

        # Basic properties
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        # FOV
        self.fov = fov
        self.uv_shape = tuple(self.fov.uv_shape.vals)

        # Axes
        self.axes = list(axes)
        self.u_axis = self.axes.index('u')
        self.v_axis = self.axes.index('v')
        self.swap_uv = (self.u_axis > self.v_axis)

        self.t_axis = -1

        # Shape / Size
        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        # Cadence
        self.cadence = Metronome.for_array0d(tstart, texp)

        # Timing
        self.tstart = self.cadence.tstart
        self.texp = self.cadence.texp

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self._scalar_time = (Scalar(self.time[0]), Scalar(self.time[1]))
        self._scalar_midtime = Scalar(self.midtime)

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        return (self.axes, self.tstart, self.texp, self.fov, self.path,
                self.frame, self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])

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

        indices = Vector.as_vector(indices, recursive=derivs)
        uv = indices.to_pair((self.u_axis, self.v_axis))
        time = self._scalar_midtime

        if remask:
            is_outside = self.uv_is_outside(uv, inclusive=True)
            if np.any(is_outside.vals):
                uv = uv.remask_or(is_outside.vals)
                time = Scalar.filled(uv.shape, self.midtime, mask=uv.mask)

        return (uv, time)

    #===========================================================================
    def uvt_range(self, indices, remask=False):
        """Ranges of (u,v) spatial coordinates and time for integer array
        indices.

        Input:
            indices     a Scalar or Vector of array indices.
            remask      True to mask values outside the field of view.

        Return:         (uv_min, uv_max, time_min, time_max)
            uv_min      a Pair defining the minimum values of FOV (u,v)
                        associated the pixel.
            uv_max      a Pair defining the maximum values of FOV (u,v)
                        associated the pixel.
            time_min    a Scalar defining the minimum time associated with the
                        array indices. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        indices = Vector.as_vector(indices, recursive=False)
        uv = indices.to_pair((self.u_axis, self.v_axis))
        uv_min = uv.int(top=self.uv_shape, remask=remask)

        # Times can be returned "shapeless" unless a mask is needed
        time_min = Scalar.filled(uv.shape, self.time[0], mask=uv_min.mask)
        time_max = Scalar.filled(uv.shape, self.time[1], mask=uv_min.mask)

        return (uv_min, uv_min + Pair.INT11, time_min, time_max)

    #===========================================================================
    def uv_range_at_tstep(self, tstep, remask=False):
        """A tuple defining the range of spatial (u,v) pixels active at a
        particular time step.

        Input:
            tstep       a Scalar time step index.
            remask      True to mask values outside the time interval.

        Return:         a tuple (uv_min, uv_max)
            uv_min      a Pair defining the minimum values of FOV (u,v)
                        coordinates active at this time step.
            uv_min      a Pair defining the maximum values of FOV (u,v)
                        coordinates active at this time step (exclusive).
        """

        uv_min = Pair.INT00     # without a mask, return shapeless pairs
        uv_max = Pair.as_pair(self.fov.uv_shape)

        # If the object needs a mask, expand it and mask it
        tstep = Scalar.as_scalar(tstep, recursive=False)
        if remask or np.any(tstep.mask):
            new_mask = Qube.or_(tstep.mask, (tstep.vals < 0) | (tstep.vals > 1))
            uv_min = Pair.zeros(tstep.shape, dtype='int', mask=new_mask)
            uv_max = Pair.filled(tstep.shape, self.uv_shape, mask=new_mask)

        return (uv_min, uv_max)

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

        uv_pair = Pair.as_pair(uv_pair)

        if remask or np.any(uv_pair.mask):
            is_outside = self.uv_is_outside(uv_pair, inclusive=True)
            new_mask = Qube.or_(is_outside.vals, uv_pair.mask)
            if new_mask is not False:
                time_min = Scalar.filled(uv_pair.shape, self.time[0],
                                                        mask=new_mask)
                time_max = Scalar.filled(uv_pair.shape, self.time[1],
                                                        mask=new_mask)
                return (time_min, time_max)

        # Without a mask, it's OK to return shapeless values
        return self._scalar_time

    #===========================================================================
    def uv_range_at_time(self, time, remask=False):
        """The (u,v) range of spatial pixels observed at the specified time.

        For the Snapshot observation subclass, the (u,v) range is always the
        same, spanning the shape of the FOV. The input time is largely ignored,
        although it is expected to fall within the time limits of the
        observation and will be masked if remask == True.

        Input:
            time        a Scalar of time values in seconds TDB.
            remask      True to mask values outside the time limits.

        Return:         (uv_min, uv_max)
            uv_min      the lower (u,v) corner Pair of the area observed at the
                        specified time.
            uv_max      the upper (u,v) corner Pair of the area observed at the
                        specified time.
        """

        return Observation.uv_range_at_time_0d(self, time,
                                                     uv_shape=self.fov.uv_shape,
                                                     remask=remask)

    #===========================================================================
    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = Snapshot(axes=self.axes, tstart=self.time[0] + dtime,
                       texp=self.texp, fov=self.fov, path=self.path,
                       frame=self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

    ############################################################################
    # Overrides of Observation methods
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
                        tfrac and time can be specified; the other must be None.
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
        velocity WRT the SSB could vary during the observation. This can be
        neglected for a Snapshot.
        """

        # Limit iterations to 1 for Snapshot
        return super(Snapshot, self).uv_from_ra_and_dec(ra, dec,
                                                        tfrac=tfrac, time=time,
                                                        apparent=apparent,
                                                        derivs=derivs,
                                                        iters=1, quick=quick)

    #===========================================================================
    def uv_from_path(self, path, tfrac=0.5, time=None, derivs=False, guess=None,
                           quick={}, converge={}):
        """The (u,v) indices of an object in the FOV, given its path.

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

        # Convert tfrac to time. That way, iteration is avoided
        if tfrac is not None:
            if time is not None:
                raise ValueError('tfrac and time cannot both be defined')

            time = self.time[0] + tfrac * (self.time[1] - self.time[0])

        return super(Snapshot, self).uv_from_path(path, tfrac=tfrac, time=time,
                                                  derivs=False, guess=None,
                                                  quick={}, converge={})

    #===========================================================================
    def uv_from_coords(self, surface, coords, tfrac=0.5, time=None,
                             underside=False, derivs=False,
                             quick={}, converge={}):
        """The (u,v) indices of a surface point, given its coordinates.

        **** NOT WELL TESTED! ****

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
                        found. The path is evaluated at the mid-time of this
                        pixel.
        """

        if tfrac is not None:
            if time is not None:
                raise ValueError('tfrac and time cannot both be defined')

            time = self.time[0] + tfrac * (self.time[1] - self.time[0])

        obs_event = Event(time, Vector3.ZERO, self.path, self.frame)
        (surface_event,
         obs_event) = surface.photon_to_event_by_coords(obs_event, coords,
                                derivs=derivs, quick=quick, converge=converge)

        neg_arr_ap = obs_event.neg_arr_ap
        if not underside:
            normal = surface.normal(surface_event.pos)
            mask = (normal.dot(surface_event.dep_ap, recursive=False) < 0.)
            neg_arr_ap = neg_arr_ap.remask_or(mask)

        return self.fov.uv_from_los_t(neg_arr_ap, time=time, derivs=derivs)

    #===========================================================================
    def inventory(self, bodies, tfrac=0.5, time=None, expand=0., cache=True,
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
                        accommodate pointing uncertainties.
            cache       if False, do not cache the body paths.  Default is True.
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

        if return_type not in ('list', 'flags', 'full'):
            raise ValueError('invalid return_type for Observation.inventory: '
                             + repr(return_type))

        if fov is None:
            fov = self.fov

        body_names = [Body.as_body_name(body) for body in bodies]
        bodies  = [Body.as_body(body) for body in bodies]
        nbodies = len(bodies)

        path_ids = [body.path for body in bodies]
        path_id = '+' if cache else None
        multipath = MultiPath(path_ids, path_id=path_id)

        if time is None:
            tfrac = Scalar.as_scalar(tfrac)
            obs_time = self.time[0] + tfrac * (self.time[1] - self.time[0])
        else:
            obs_time = Scalar.as_scalar(time)

        obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)
        (_,
         arrival_event) = multipath.photon_to_event(obs_event, quick=quick,
                                                    converge=converge)

        centers = arrival_event.neg_arr_ap
        ranges = centers.norm()
        radii = Scalar([body.radius for body in bodies])
        radius_angles = (radii / ranges).arcsin()

        inner_radii = Scalar([body.inner_radius for body in bodies])
        inner_angles = (inner_radii / ranges).arcsin()

        # This array equals True for each body falling somewhere inside the FOV
        falls_inside = np.empty(nbodies, dtype='bool')
        for i in range(nbodies):
            falls_inside[i] = fov.sphere_falls_inside(centers[i], radii[i],
                                                      time=obs_time, border=expand)

        # This array equals True for each body completely hidden by another
        is_hidden = np.zeros(nbodies, dtype='bool')
        for i in range(nbodies):
            if not falls_inside[i]:
                continue

            for j in range(nbodies):
                if not falls_inside[j]:
                    continue

                if ranges[i] < ranges[j]:
                    continue
                if radius_angles[i] > inner_angles[j]:
                    continue

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
                if flags[i]:
                    ret_list.append(body_names[i])
            return ret_list

        # Return full info
        returned_dict = {}

        u_scale = fov.uv_scale.vals[0]
        v_scale = fov.uv_scale.vals[1]
        body_uv = fov.uv_from_los_t(arrival_event.neg_arr_ap,
                                    time=obs_time).vals
        for i in range(nbodies):
            body_data = {}
            body_data['name'] = body_names[i]
            body_data['inside'] = flags[i]
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
            u_min_unclipped = int(np.floor(u-radius_angles[i].vals/u_scale))
            u_max_unclipped = int(np.ceil( u+radius_angles[i].vals/u_scale))
            v_min_unclipped = int(np.floor(v-radius_angles[i].vals/v_scale))
            v_max_unclipped = int(np.ceil( v+radius_angles[i].vals/v_scale))

            body_data['u_min_unclipped'] = u_min_unclipped
            body_data['u_max_unclipped'] = u_max_unclipped
            body_data['v_min_unclipped'] = v_min_unclipped
            body_data['v_max_unclipped'] = v_max_unclipped

            body_data['u_min'] = np.clip(u_min_unclipped, 0, self.uv_shape[0]-1)
            body_data['u_max'] = np.clip(u_max_unclipped, 0, self.uv_shape[0]-1)
            body_data['v_min'] = np.clip(v_min_unclipped, 0, self.uv_shape[1]-1)
            body_data['v_max'] = np.clip(v_max_unclipped, 0, self.uv_shape[1]-1)

            body_data['u_pixel_size'] = radius_angles[i].vals/u_scale*2
            body_data['v_pixel_size'] = radius_angles[i].vals/v_scale*2

            returned_dict[body_names[i]] = body_data

        return returned_dict

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Snapshot(unittest.TestCase):

    def runTest(self):

        from oops.fov.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (10,20))
        obs = Snapshot(('u','v'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0.,0.),(0.,20.),(10.,0.),(10.,20.),(10.,21.)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

        # uvt() with remask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(time.mask)
        self.assertEqual(time, 99.)
        self.assertEqual(uv, Pair.as_pair(indices))

        # uvt() with remask == True
        (uv,time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(4*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:4], 99.)
        self.assertEqual(uv[:4], Pair.as_pair(indices)[:4])

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, Pair.as_pair(indices_))
        self.assertEqual(uv_max, Pair.as_pair(indices_) + (1,1))
        self.assertEqual(time_min,  98.)
        self.assertEqual(time_max, 100.)

        # uvt_range() with remask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min,  98.)
        self.assertEqual(time_max, 100.)

        # uvt_range() with remask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                             remask=True)
        self.assertTrue(np.all(uv_min.mask == [False] + 4*[True]))
        self.assertTrue(np.all(uv_min.mask == uv_max.mask))
        self.assertTrue(np.all(uv_min.mask == time_min.mask))
        self.assertTrue(np.all(uv_min.mask == time_max.mask))

        self.assertEqual(uv_min[0], Pair.as_pair(indices)[0])
        self.assertEqual(uv_max[0], (Pair.as_pair(indices) + (1,1))[0])
        self.assertEqual(time_min[0],  98.)
        self.assertEqual(time_max[0], 100.)

        # time_range_at_uv() with remask == False
        uv_pair = Pair([(0.,0.),(0.,20.),(10.,0.),(10.,20.),(10.,21.)])

        (time0, time1) = obs.time_range_at_uv(uv_pair)

        self.assertEqual(time0,  98.)
        self.assertEqual(time1, 100.)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv_pair, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4],  98.)
        self.assertEqual(time1[:4], 100.)

        # Alternative axis order ('v','u')
        obs = Snapshot(('v','u'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(20,0),(20,10),(20,11)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, indices.to_pair((1,0)))

        (uv,time) = obs.uvt(indices, remask=True)

        self.assertEqual(uv[:4], indices.to_pair((1,0))[:4])
        self.assertTrue(np.all(uv.mask == 4*[False] + [True]))

        # Alternative axis order ('v', 'a', 'u')
        obs = Snapshot(('v','a','u'), tstart=98., texp=2.,
                       fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,-1,0),(0,99,10),(20,-9,0),(20,77,10),(20,44,11)])
        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, indices.to_pair((2,0)))

        (uv,time) = obs.uvt(indices, remask=True)

        self.assertEqual(uv[:4], indices.to_pair((2,0))[:4])
        self.assertTrue(np.all(uv.mask == 4*[False] + [True]))

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
