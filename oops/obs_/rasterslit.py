################################################################################
# oops/obs_/rasterslit.py: Subclass RasterSlit of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation import Observation
from oops.path_.path       import Path
from oops.frame_.frame     import Frame
from oops.event            import Event

class RasterSlit(Observation):
    """A RasterSlit is subclass of Observation consisting of a 2-D image in
    which one dimension is constructed by sweeping a single pixel along a slit,
    and the second dimension is simulated by rotation of the camera. The FOV
    describes the 1-D slit. This differs from a Slit subclass in that a single
    sensor is moving to emulate the 1-D slit. It differs from a RasterScan in
    that the FOV describes a 1-D slit and not a 2-D image.
    """

    PACKRAT_ARGS = ['axes', 'det_size', 'cadence', 'fov', 'path', 'frame',
                    '**subfields']

    def __init__(self, axes, det_size, cadence, fov, path, frame, **subfields):
        """Constructor for a RasterSlit observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'ufast' or
                        'uslow' should appear at the location of the array's
                        u-axis; 'vslow' or 'vfast' should appear at the location
                        of the array's v-axis. The 'fast' suffix identifies
                        which of these is in the fast-scan direction; the 'slow'
                        suffix identifies the slow-scan direction.
            det_size    the size of the detector in FOV units parallel to the
                        slit. It will be < 1 if there are gaps between the
                        samples, or > 1 if the detector moves by less than its
                        full size within the fast time step.

            cadence     a 2-D Cadence object defining the timing of each
                        consecutive measurement. The first index defines time
                        sampling along the slow axis and the second defines
                        time sub-sampling along the fast axis, which corresponds
                        to the motion of the detector within the slit.
            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a RasterSlit object, one of the
                        axes of the FOV must have length 1.
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

        assert len(self.cadence.shape) == 2

        self.axes = list(axes)
        assert (('ufast' in self.axes and 'vslow' in self.axes) or
                ('vfast' in self.axes and 'uslow' in self.axes))

        if 'ufast' in self.axes:
            self.u_axis = self.axes.index('ufast')
            self.v_axis = self.axes.index('vslow')
            self.fast_axis = self.u_axis
            self.slow_axis = self.v_axis
            self.cross_slit_uv_axis = 1
            self.along_slit_uv_axis = 0
            self.uv_shape = [self.cadence.shape[1], self.cadence.shape[0]]
        else:
            self.u_axis = self.axes.index('uslow')
            self.v_axis = self.axes.index('vfast')
            self.fast_axis = self.v_axis
            self.slow_axis = self.u_axis
            self.cross_slit_uv_axis = 0
            self.along_slit_uv_axis = 1
            self.uv_shape = [self.cadence.shape[0], self.cadence.shape[1]]

        self.swap_uv = (self.u_axis > self.v_axis)

        self.along_slit_shape = self.uv_shape[self.along_slit_uv_axis]

        self.t_axis = [self.slow_axis, self.fast_axis]
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        assert self.fov.uv_shape.vals[self.cross_slit_uv_axis] == 1
        assert (self.fov.uv_shape.vals[self.along_slit_uv_axis] ==
                self.cadence.shape[1])

        self.det_size = det_size
        self.slit_is_discontinuous = (self.det_size != 1)

        duv_dt_basis_vals = np.zeros(2)
        duv_dt_basis_vals[self.along_slit_uv_axis] = 1.
        self.duv_dt_basis = Pair(duv_dt_basis_vals)

        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def uvt(self, indices, fovmask=False):
        """Return coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Vector (or subclass) of array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        indices = Vector.as_vector(indices)
        slit_coord = indices.to_scalar(self.fast_axis)

        # Handle discontinuous detectors
        if self.slit_is_discontinuous:

            # Identify indices at exact upper limit; treat these as inside
            at_upper_limit = (slit_coord == self.along_slit_shape)

            # Map continuous index to discontinuous (u,v)
            slit_int = slit_coord.int()
            slit_coord = slit_int + (slit_coord - slit_int) * self.det_size

            # Adjust values at upper limit
            slit_coord = slit_coord.mask_where(at_upper_limit,
                            replace = self.along_slit_shape + self.det_size - 1,
                            remask = False)

        # Create (u,v) Pair
        uv_vals = np.empty(indices.shape + (2,))
        uv_vals[..., self.along_slit_uv_axis] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_axis] = 0.5
        uv = Pair(uv_vals, indices.mask)

        # Create time Scalar
        tstep = indices.to_pair(self.t_axis)
        time = self.cadence.time_at_tstep(tstep, mask=fovmask)

        # Apply mask if necessary
        if fovmask:
            u_index = indices.vals[..., self.u_axis]
            v_index = indices.vals[..., self.v_axis]
            is_outside = ((u_index < 0) |
                          (v_index < 0) |
                          (u_index > self.uv_shape[0]) |
                          (v_index > self.uv_shape[1]))
            if np.any(is_outside):
                uv = uv.mask_where(is_outside)
                time = time.mask_where(is_outside)

        return (uv, time)

    def uvt_range(self, indices, fovmask=False):
        """Return ranges of coordinates and time for integer array indices.

        Input:
            indices     a Vector (or subclass) of integer array indices.
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

        slit_coord = indices.to_scalar(self.fast_axis)

        uv_vals = np.empty(indices.shape + (2,), dtype='int')
        uv_vals[..., self.along_slit_uv_axis] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_axis] = 0
        uv_min = Pair(uv_vals, indices.mask)
        uv_max = uv_min + Pair.ONES

        tstep = indices.to_pair(self.t_axis)
        (time_min,
         time_max) = self.cadence.time_range_at_tstep(tstep, mask=fovmask)

        if fovmask:
            u_index = indices.vals[..., self.u_axis]
            v_index = indices.vals[..., self.v_axis]
            is_outside = ((u_index < 0) |
                          (v_index < 0) |
                          (u_index >= self.uv_shape[0]) |
                          (v_index >= self.uv_shape[1]))

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
            tstep       a time step index (one or two integers). Not checked for
                        out-of-range errors.

        Return:         a tuple (uv_min, uv_max)
            uv_min      a Pair defining the minimum values of (u,v) coordinates
                        active at this time step.
            uv_min      a Pair defining the maximum values of (u,v) coordinates
                        active at this time step (exclusive).
        """

        if self.along_slit_uv_axis == 0:
            return (Pair(tstep[1], 0), Pair(tstep[1]+1, 1))
        else:
            return (Pair(0, tstep[1]), Pair(1, tstep[1]+1))

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
        tstep = uv_pair.to_pair((self.cross_slit_uv_axis,
                                 self.along_slit_uv_axis))

        return self.cadence.time_range_at_tstep(tstep, mask=fovmask)

    def sweep_duv_dt(self, uv_pair):
        """Return the mean local sweep speed of the instrument along (u,v) axes.

        Input:
            uv_pair     a Pair of spatial indices (u,v).

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        uv_pair = Pair.as_pair(uv_pair)
        tstep = uv_pair.to_scalar(self.cross_slit_uv_axis)

        return self.duv_dt_basis / self.cadence.tstride_at_tstep(tstep)

    def time_shift(self, dtime):
        """Return a copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = RasterSlit(self.axes, self.uv_size,
                         self.cadence.time_shift(dtime),
                         self.fov, self.path, self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_RasterSlit(unittest.TestCase):

    def runTest(self):

        from oops.cadence_.metronome import Metronome
        from oops.cadence_.dual import DualCadence
        from oops.fov_.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), det_size=1,
                         cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        # uvt() with fovmask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, slow_cadence.tstride * indices.to_scalar(1) +
                               fast_cadence.tstride * indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        # uvt() with fovmask == True
        (uv, time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6],
                         (slow_cadence.tstride * indices.to_scalar(1) +
                          fast_cadence.tstride * indices.to_scalar(0))[:6])
        self.assertEqual(uv[:6].to_scalar(0), indices[:6].to_scalar(0))
        self.assertEqual(uv[:6].to_scalar(1), 0.5)

        # uvt() with fovmask == True, new indices
        (uv, time) = obs.uvt(indices+(0.2,0.9), fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(time.mask == uv.mask))

        # uvt_range() with fovmask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min,
                         slow_cadence.tstride * indices.to_scalar(1) +
                         fast_cadence.tstride * indices.to_scalar(0))
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        # uvt_range() with fovmask == True
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.to_scalar(0)[:2], indices.to_scalar(0)[:2])
        self.assertEqual(uv_min.to_scalar(1)[:2], 0)
        self.assertEqual(uv_max.to_scalar(0)[:2], indices.to_scalar(0)[:2] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:2], 1)
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

        ####################################
        # Alternative axis order ('uslow','vfast')

        fov = FlatFOV((0.001,0.001), (1,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        fast_cadence = Metronome(tstart=0., tstride=0.5, texp=0.5, steps=20)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('uslow','vfast'), det_size=1,
                         cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(time, slow_cadence.tstride * indices.to_scalar(0) +
                               fast_cadence.tstride * indices.to_scalar(1))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices.to_scalar(1) + 1)
        self.assertEqual(time_min, slow_cadence.tstride * indices.to_scalar(0) +
                                   fast_cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        (time0,time1) = obs.times_at_uv(indices)

        self.assertEqual(time0, slow_cadence.tstride * indices.to_scalar(0) +
                                fast_cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time1, time0 + fast_cadence.texp)

        ####################################
        # Alternative det_size and texp for discontinuous indices

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=8., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.5, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), det_size=0.5,
                         cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 199.5)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1],  5.)
        self.assertEqual(obs.uvt((5,5))[1], 55.)
        self.assertEqual(obs.uvt((5.0, 5.5))[1], 55.)
        self.assertEqual(obs.uvt((5.5, 5.0))[1], 55.25)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6.     ,0))[1] - 6.000) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 6.125) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 6.250) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 6.375) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 6.500) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 7.000) < delta)

        self.assertEqual(obs.uvt((0,0))[0], (0.,0.5))
        self.assertEqual(obs.uvt((5,0))[0], (5.,0.5))
        self.assertEqual(obs.uvt((5,5))[0], (5.,0.5))

        self.assertTrue(abs(obs.uvt((6.     ,0))[0] - (6.0,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (6.1,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (6.2,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (6.3,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (6.4,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,5))[0] - (6.5,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (7.0,0.5)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ))[0] - (1.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((2, 1.25   ))[0] - (2.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((3, 2.5    ))[0] - (3.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((4, 3.75   ))[0] - (4.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5 - eps))[0] - (5.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5.     ))[0] - (5.,0.5)) < delta)

        # Alternative tstride for even more discontinuous indices
        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=11., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), det_size=0.5,
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
        pair = (10-eps,0)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[0] - 9.5) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10,0)
        self.assertTrue(abs(obs.uvt(pair, True)[0].values[0] -  9.5) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10+eps,0)
        self.assertTrue(obs.uvt(pair, True)[0].mask)

        # Try all at once
        indices = Pair([(10-eps,0), (10,0), (10+eps,0)])

        (uv,t) = obs.uvt(indices, fovmask=True)
        self.assertTrue(np.all(t.mask == np.array(2*[False] + [True])))

        # Alternative with uv_size and texp and axes
        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('a','vslow','b','ufast','c'), det_size=0.5,
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

        self.assertEqual(obs.uvt((0,0,0,0,0))[0], (0.,0.5))
        self.assertEqual(obs.uvt((0,0,0,5,0))[0], (5.,0.5))
        self.assertEqual(obs.uvt((0,5,0,5,0))[0], (5.,0.5))

        self.assertTrue(abs(obs.uvt((1,0,4,6      ,7))[0] - (6.0,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,1,4,6.2    ,7))[0] - (6.1,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,2,4,6.4    ,7))[0] - (6.2,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,3,4,6.6    ,7))[0] - (6.3,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,4,4,6.8    ,7))[0] - (6.4,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,5,4,7 - eps,7))[0] - (6.5,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,6,4,7.     ,7))[0] - (7.0,0.5)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ,4,1,7))[0] - (1.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 1.25   ,4,2,7))[0] - (2.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (3.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 3.75   ,4,4,7))[0] - (4.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5 - eps,4,5,7))[0] - (5.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5.     ,4,5,7))[0] - (5.,0.5)) < delta)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
