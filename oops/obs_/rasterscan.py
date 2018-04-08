################################################################################
# oops/obs_/rasterscan.py: Subclass RasterScan of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation import Observation
from oops.path_.path       import Path
from oops.path_.multipath  import MultiPath
from oops.frame_.frame     import Frame
from oops.body             import Body
from oops.event            import Event

class RasterScan(Observation):
    """A RasterScan is subclass of Observation consisting of a 2-D image
    generated by sweeping a single sensor within a 2-D field of view.

    The FOV object defines the entire field of view, although each pixel is
    sampled at a different time step. The sampling time of each pixel is defined
    by a 2-D cadence.
    """

    INVENTORY_IMPLEMENTED = True

    PACKRAT_ARGS = ['axes', 'uv_size', 'cadence', 'fov', 'path', 'frame',
                    '**subfields']

    def __init__(self, axes, uv_size, cadence, fov, path, frame, **subfields):
        """Constructor for a RasterScan observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'ufast' or
                        'uslow' should appear at the location of the array's
                        u-axis; 'vslow' or 'vfast' should appear at the location
                        of the array's v-axis. The 'fast' suffix identifies
                        which of these is in the fast-scan direction; the 'slow'
                        suffix identifies the slow-scan direction.
            uv_size     the size of the detector in units of the FOV along the
                        (u,v) axes. A value of (1,1) would indicate that there
                        is no dead space between the detectors. A value < 1
                        indicates a gap along that axis; a value > 1 indicates
                        that the detector is larger than the shift, yielding
                        overlaps.

            cadence     a 2-D Cadence object defining the start time and
                        duration of each sample.
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

        self.cadence = cadence
        self.fov = fov
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        self.axes = list(axes)
        assert (('ufast' in self.axes and 'vslow' in self.axes) or
                ('vfast' in self.axes and 'uslow' in self.axes))

        if 'ufast' in self.axes:
            self.u_axis = self.axes.index('ufast')
            self.v_axis = self.axes.index('vslow')
            self.fast_axis = self.u_axis
            self.slow_axis = self.v_axis
            self.fast_uv_axis = 0
            self.slow_uv_axis = 1
        else:
            self.u_axis = self.axes.index('uslow')
            self.v_axis = self.axes.index('vfast')
            self.fast_axis = self.v_axis
            self.slow_axis = self.u_axis
            self.fast_uv_axis = 1
            self.slow_uv_axis = 0

        self.swap_uv = (self.u_axis > self.v_axis)

        self.t_axis = [self.slow_axis, self.fast_axis]
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self.uv_shape = tuple(self.fov.uv_shape.vals)
        assert len(self.cadence.shape) == 2
        assert self.cadence.shape[0] == self.uv_shape[self.slow_uv_axis]
        assert self.cadence.shape[1] == self.uv_shape[self.fast_uv_axis]

        self.uv_size = Pair.as_pair(uv_size)
        self.uv_is_discontinuous = (self.uv_size != Pair.ONES)

        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        return

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

        indices = Vector.as_vector(indices)
        uv = indices.to_pair((self.u_axis,self.v_axis))

        # Handle discontinuous detectors
        if self.uv_is_discontinuous:

            # Identify indices at exact upper limits; treat these as inside
            at_upper_u = (uv.values[...,0] == self.uv_shape[0])
            at_upper_v = (uv.values[...,1] == self.uv_shape[1])

            # Map continuous index to discontinuous (u,v)
            uv_int = Pair.as_pair(uv).as_int()
            uv = uv_int + (uv - uv_int).element_mul(self.uv_size)

            # Adjust values at upper limits
            u = uv.to_scalar(0).mask_where(at_upper_u,
                    replace = self.uv_shape[0] + self.uv_size.values[0] - 1,
                    remask = False)
            v = uv.to_scalar(1).mask_where(at_upper_v,
                    replace = self.uv_shape[1] + self.uv_size.values[1] - 1,
                    remask = False)

            # Re-create Pair
            uv_values = np.empty(u.shape + (2,))
            uv_values[...,0] = u.values
            uv_values[...,1] = v.values

            uv = Pair(uv_values, indices.mask)

        # Create the time Scalar
        tstep = indices.to_pair(self.t_axis)
        time = self.cadence.time_at_tstep(tstep, mask=fovmask)

        # Apply mask if necessary
        if fovmask:
            is_outside = self.uv_is_outside(uv, inclusive=True)
            if np.any(is_outside):
                uv = uv.mask_where(is_outside)
                time = time.mask_where(is_outside)

        return (uv, time)

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

        indices = Vector.as_vector(indices).as_int()

        uv_min = indices.to_pair((self.u_axis,self.v_axis))
        uv_max = uv_min + self.uv_size

        tstep = indices.to_pair(self.t_axis)
        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep,
                                                                mask=fovmask)

        if fovmask:
            is_outside = self.uv_is_outside(uv_min, inclusive=False)
            if np.any(is_outside):
                uv_min = uv_min.mask_where(is_outside)
                uv_max = uv_max.mask_where(is_outside)
                time_min = time_min.mask_where(is_outside)
                time_max = time_max.mask_where(is_outside)

        return (uv_min, uv_max, time_min, time_max)

    def uv_range_at_tstep(self, *tstep):
        """Return a tuple defining the range of (u,v) coordinates active at a
        particular time step.

        Input:
            tstep       a time step index (one or two integers).

        Return:         a tuple (uv_min, uv_max)
            uv_min      a Pair defining the minimum values of (u,v) coordinates
                        active at this time step.
            uv_min      a Pair defining the maximum values of (u,v) coordinates
                        active at this time step (exclusive).
        """

        if self.fast_uv_axis == 0:
            return (Pair(tstep[1], tstep[0]), Pair(tstep[1]+1, tstep[0]+1))
        else:
            return (Pair(tstep[0], tstep[1]), Pair(tstep[0]+1, tstep[1]+1))

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

        uv_pair = Pair.as_pair(uv_pair).as_int()
        tstep = uv_pair.to_pair((self.slow_uv_axis, self.fast_uv_axis))

        return self.cadence.time_range_at_tstep(tstep, mask=fovmask)

    def sweep_duv_dt(self, uv_pair):
        """Return the mean local sweep speed of the instrument along (u,v) axes.

        Input:
            uv_pair     a Pair of spatial indices (u,v).

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        uv_pair = Pair.as_pair(uv_pair).as_int()
        tstep = uv_pair.as_pair((self.slow_uv_axis, self.fast_uv_axis))

        return Pair.ONES / self.cadence.tstride_at_tstep(tstep)

    def time_shift(self, dtime):
        """Return a copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = RasterScan(self.axes, self.uv_size,
                         self.cadence.time_shift(dtime),
                         self.fov, self.path, self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

    def inventory(self, bodies, expand=0., return_type='list', fov=None,
                        quick={}, converge={}, time_frac=0.5):
        """Return the body names that appear unobscured inside the FOV.

        WARNING: Not properly updated for class RasterScan. Use at your own risk.

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
        returned_dict = {}

        u_scale = fov.uv_scale.vals[0]
        v_scale = fov.uv_scale.vals[1]
        body_uv = fov.uv_from_los(arrival_event.neg_arr_ap).vals
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

class Test_RasterScan(unittest.TestCase):

    def runTest(self):

        from oops.cadence_.metronome import Metronome
        from oops.cadence_.dual import DualCadence
        from oops.fov_.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('ufast','vslow'), uv_size=(1,1),
                         cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        # uvt() with fovmask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, slow_cadence.tstride * indices.to_scalar(1) +
                               fast_cadence.tstride * indices.to_scalar(0))
        self.assertEqual(uv, Pair.as_pair(indices))

        # uvt() with fovmask == True
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6],
                         (slow_cadence.tstride * indices.to_scalar(1) +
                          fast_cadence.tstride * indices.to_scalar(0))[:6])
        self.assertEqual(uv[:6], Pair.as_pair(indices)[:6])

        # uvt_range() with fovmask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min,
                         slow_cadence.tstride * indices.to_scalar(1) +
                         fast_cadence.tstride * indices.to_scalar(0))
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        # uvt_range() with fovmask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min, slow_cadence.tstride * indices.to_scalar(1) +
                                   fast_cadence.tstride * indices.to_scalar(0))
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        # uvt_range() with fovmask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], Pair.as_pair(indices)[:2])
        self.assertEqual(uv_max[:2], Pair.as_pair(indices)[:2] + (1,1))
        self.assertEqual(time_min[:2],
                         (slow_cadence.tstride * indices.to_scalar(1) +
                          fast_cadence.tstride * indices.to_scalar(0))[:2])
        self.assertEqual(time_max[:2], time_min[:2] + fast_cadence.texp)

        # times_at_uv() with fovmask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, slow_cadence.tstride * uv.to_scalar(1) +
                                fast_cadence.tstride * uv.to_scalar(0))
        self.assertEqual(time1, time0 + fast_cadence.texp)

        # times_at_uv() with fovmask == True
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4],
                         (slow_cadence.tstride * uv.to_scalar(1) +
                          fast_cadence.tstride * uv.to_scalar(0))[:4])
        self.assertEqual(time1[:4], time0[:4] + fast_cadence.texp)

        # Alternative tstride (10,1)
        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=20)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('uslow','vfast'), uv_size=(1,1),
                         cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, Pair.as_pair(indices))
        self.assertEqual(time, slow_cadence.tstride * indices.to_scalar(0) +
                               fast_cadence.tstride * indices.to_scalar(1))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min, slow_cadence.tstride * indices.to_scalar(0) +
                                   fast_cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        (time0,time1) = obs.times_at_uv(indices)

        self.assertEqual(time0, slow_cadence.tstride * indices.to_scalar(0) +
                                fast_cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time1, time0 + fast_cadence.texp)

        # Alternative uv_size and texp for discontinuous indices
        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('ufast','vslow'), uv_size=(0.5,0.8),
                         cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 199.8)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1],  5.)
        self.assertEqual(obs.uvt((5,5))[1], 55.)
        self.assertEqual(obs.uvt((5.0, 5.5))[1], 55.)
        self.assertEqual(obs.uvt((5.5, 5.0))[1], 55.4)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6.     ,0))[1] - 6. ) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 6.2) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 6.4) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 6.6) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 6.8) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 7.0) < delta)

        self.assertEqual(obs.uvt((0,0))[0], (0.,0.))
        self.assertEqual(obs.uvt((5,0))[0], (5.,0.))
        self.assertEqual(obs.uvt((5,5))[0], (5.,5.))

        self.assertTrue(abs(obs.uvt((6.     ,0))[0] - (6.0,0.)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (6.1,1.)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (6.2,2.)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (6.3,3.)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (6.4,4.)) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,5))[0] - (6.5,5.)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (7.0,6.)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ))[0] - (1.,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((2, 1.25   ))[0] - (2.,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((3, 2.5    ))[0] - (3.,2.4)) < delta)
        self.assertTrue(abs(obs.uvt((4, 3.75   ))[0] - (4.,3.6)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5 - eps))[0] - (5.,4.8)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5.     ))[0] - (5.,5.0)) < delta)

        # Alternative tstride for even more discontinuous indices
        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=11., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('ufast','vslow'), uv_size=(0.5,0.8),
                         cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 218.8)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1],  5.)
        self.assertEqual(obs.uvt((5,5))[1], 60.)
        self.assertEqual(obs.uvt((5.0, 5.5))[1], 60.)
        self.assertEqual(obs.uvt((5.5, 5.0))[1], 60.4)
        self.assertEqual(obs.uvt((5.5, 5.5))[1], 60.4)

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6      ,0))[1] - 6. ) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 6.2) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 6.4) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 6.6) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 6.8) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 7.0) < delta)

        self.assertTrue(abs(obs.uvt((9       ,0))[1] -  9. ) < delta)
        self.assertTrue(abs(obs.uvt((9.25    ,0))[1] -  9.2) < delta)
        self.assertTrue(abs(obs.uvt((9.5     ,0))[1] -  9.4) < delta)
        self.assertTrue(abs(obs.uvt((9.75    ,0))[1] -  9.6) < delta)
        self.assertTrue(abs(obs.uvt((10 - eps,0))[1] -  9.8) < delta)
        self.assertTrue(abs(obs.uvt((0.      ,1))[1] - 11. ) < delta)

        self.assertTrue(abs(obs.uvt((6.00, 0.   ))[1] -  6. ) < delta)
        self.assertTrue(abs(obs.uvt((6.25, 0.   ))[1] -  6.2) < delta)
        self.assertTrue(abs(obs.uvt((6.25, 1.   ))[1] - 17.2) < delta)
        self.assertTrue(abs(obs.uvt((6.25, 2-eps))[1] - 17.2) < delta)
        self.assertTrue(abs(obs.uvt((6.25, 2    ))[1] - 28.2) < delta)

        # Test the upper edge
        pair = (10-eps,20-eps)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[0] -  9.5) < delta)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[1] - 19.8) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10,20-eps)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[0] -  9.5) < delta)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[1] - 19.8) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10-eps,20)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[0] -  9.5) < delta)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[1] - 19.8) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10,20)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[0] -  9.5) < delta)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[1] - 19.8) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        self.assertTrue(obs.uvt((10+eps,20), True)[0].mask)
        self.assertTrue(obs.uvt((10,20+eps), True)[0].mask)

        # Try all at once
        indices = Pair([(10-eps,20-eps), (10,20-eps), (10-eps,20), (10,20),
                        (10+eps,20), (10,20+eps)])

        (uv,t) = obs.uvt(indices, fovmask=True)
        self.assertTrue(np.all(t.mask == np.array(4*[False] + 2*[True])))

        # Alternative with uv_size and texp and axes
        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('a','vslow','b','ufast','c'), uv_size=(0.5,0.8),
                         cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 199.8)

        self.assertEqual(obs.uvt((1,0,3,0,4))[1],   0.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1],   5.)
        self.assertEqual(obs.uvt((1,0,3,5.5,4))[1], 5.4)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((1,0,0,6      ,0))[1] - 6. ) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.25   ,0))[1] - 6.2) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.5    ,0))[1] - 6.4) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.75   ,0))[1] - 6.6) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7 - eps,0))[1] - 6.8) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7.     ,0))[1] - 7.0) < delta)

        self.assertEqual(obs.uvt((0,0,0,0,0))[0], (0.,0.))
        self.assertEqual(obs.uvt((0,0,0,5,0))[0], (5.,0.))
        self.assertEqual(obs.uvt((0,5,0,5,0))[0], (5.,5.))

        self.assertTrue(abs(obs.uvt((1,0,4,6      ,7))[0] - (6.0,0.)) < delta)
        self.assertTrue(abs(obs.uvt((1,1,4,6.2    ,7))[0] - (6.1,1.)) < delta)
        self.assertTrue(abs(obs.uvt((1,2,4,6.4    ,7))[0] - (6.2,2.)) < delta)
        self.assertTrue(abs(obs.uvt((1,3,4,6.6    ,7))[0] - (6.3,3.)) < delta)
        self.assertTrue(abs(obs.uvt((1,4,4,6.8    ,7))[0] - (6.4,4.)) < delta)
        self.assertTrue(abs(obs.uvt((1,5,4,7 - eps,7))[0] - (6.5,5.)) < delta)
        self.assertTrue(abs(obs.uvt((1,6,4,7.     ,7))[0] - (7.0,6.)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ,4,1,7))[0] - (1.,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((1, 1.25   ,4,2,7))[0] - (2.,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (3.,2.4)) < delta)
        self.assertTrue(abs(obs.uvt((1, 3.75   ,4,4,7))[0] - (4.,3.6)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5 - eps,4,5,7))[0] - (5.,4.8)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5.     ,4,5,7))[0] - (5.,5.0)) < delta)


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
