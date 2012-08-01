################################################################################
# oops_/obs/rasterslit.py: Subclass RasterSlit of class Observation
#
# 6/13/12 MRS - Created.
################################################################################

import numpy as np

from oops_.obs.observation_ import Observation
from oops_.array.all import *

class RasterSlit(Observation):
    """A RasterSlit is subclass of Observation consisting of a 2-D image in
    which one dimension is constructed by sweeping a single pixel along a slit,
    and the second dimension is simulated by rotation of the camera. This
    differs from a Slit subclass in that a single sensor is moving to emulate a
    1-D slit. It differs from a RasterScan in that there is no movement of the
    line of sight within the instrument's frame.
    """

    def __init__(self, axes, det_size,
                       cadence, fov, path_id, frame_id, **subfields):
        """Constructor for a RasterSlit observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of "ufast" or
                        "uslow" should appear at the location of the array's
                        u-axis; "vslow" or "vfast" should appear at the location
                        of the array's v-axis. The "fast" suffix identifies
                        which of these is in the fast-scan direction; the "slow"
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
            path_id     the registered ID of a path co-located with the
                        instrument.
            frame_id    the registered ID of a coordinate frame fixed to the
                        optics of the instrument. This frame should have its
                        Z-axis pointing outward near the center of the line of
                        sight, with the X-axis pointing rightward and the y-axis
                        pointing downward.
            subfields   a dictionary containing all of the optional attributes.
                        Additional subfields may be included as needed.
        """

        self.cadence = cadence
        self.fov = fov
        self.path_id = path_id
        self.frame_id = frame_id

        assert len(self.cadence.shape) == 2

        self.axes = list(axes)
        assert (("ufast" in self.axes and "vslow" in self.axes) or
                ("vfast" in self.axes and "uslow" in self.axes))

        if "ufast" in self.axes:
            self.u_axis = self.axes.index("ufast")
            self.v_axis = self.axes.index("vslow")
            self.fast_axis = self.u_axis
            self.slow_axis = self.v_axis
            self.cross_slit_uv_index = 1
            self.along_slit_uv_index = 0
            self.uv_shape = [self.cadence.shape[1], self.cadence.shape[0]]
        else:
            self.u_axis = self.axes.index("uslow")
            self.v_axis = self.axes.index("vfast")
            self.fast_axis = self.v_axis
            self.slow_axis = self.u_axis
            self.cross_slit_uv_index = 0
            self.along_slit_uv_index = 1
            self.uv_shape = [self.cadence.shape[0], self.cadence.shape[1]]

        self.t_axis = [self.slow_axis, self.fast_axis]
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        assert self.fov.uv_shape.vals[self.cross_slit_uv_index] == 1
        assert (self.fov.uv_shape.vals[self.along_slit_uv_index] ==
                self.cadence.shape[1])

        self.det_size = det_size
        self.slit_is_discontinuous = (self.det_size != 1)

        duv_dt_basis_vals = np.zeros(2)
        duv_dt_basis_vals[self.along_slit_uv_index] = 1.
        self.duv_dt_basis = Pair(duv_dt_basis_vals)

        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        return

    def uvt(self, indices, fovmask=False):
        """Returns the FOV coordinates (u,v) and the time in seconds TDB
        associated with the given indices into the data array. This method
        supports non-integer index values.

        Input:
            indices     a Tuple of array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        indices = Tuple.as_tuple(indices)

        slit_coord = indices.as_scalar(self.fast_axis)
        if self.slit_is_discontinuous:
            slit_int = slit_coord.int()
            slit_coord = slit_int + (slit_coord - slit_int) * self.det_size

        uv_vals = np.empty(indices.shape + [2])
        uv_vals[..., self.along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_index] = 0.5
        uv = Pair(uv_vals, indices.mask)

        tstep = indices.as_pair(self.t_axis)
        time = self.cadence.time_at_tstep(tstep)

        if fovmask:
            u_index = indices.vals[..., self.u_axis]
            v_index = indices.vals[..., self.v_axis]
            is_inside = ((u_index >= 0) &
                         (v_index >= 0) &
                         (u_index <= self.uv_shape[0]) &
                         (v_index <= self.uv_shape[1]))
            if not np.all(is_inside):
                mask = indices.mask | ~is_inside
                uv.mask = mask
                time.mask = mask

        return (uv, time)

    def uvt_range(self, indices, fovmask=False):
        """Returns the ranges of FOV coordinates (u,v) and the time range in
        seconds TDB associated with the given integer indices into the data
        array.

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

        indices = Tuple.as_int(indices)

        slit_coord = indices.as_scalar(self.fast_axis)

        uv_vals = np.empty(indices.shape + [2], dtype="int")
        uv_vals[..., self.along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_index] = 0
        uv_min = Pair(uv_vals, indices.mask)
        uv_max = uv_min + Pair.ONES

        tstep = indices.as_pair(self.t_axis)
        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep)

        if fovmask:
            u_index = indices.vals[..., self.u_axis]
            v_index = indices.vals[..., self.v_axis]
            is_inside = ((u_index >= 0) &
                         (v_index >= 0) &
                         (u_index < self.uv_shape[0]) &
                         (v_index < self.uv_shape[1]))
            if not np.all(is_inside):
                mask = indices.mask | ~is_inside
                uv_min.mask = mask
                uv_max.mask = mask
                time_min.mask = mask
                time_max.mask = mask

        return (uv_min, uv_max, time_min, time_max)

    def times_at_uv(self, uv_pair, fovmask=False, extras=None):
        """Returns the start and stop times of the specified spatial pixel
        (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.
            extras      an optional tuple or dictionary containing any extra
                        parameters required for the conversion from (u,v) to
                        time.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """

        uv_tuple = Pair.as_int(uv_pair).as_tuple()
        tstep = uv_tuple.as_pair(self.t_axis)

        return self.cadence.time_range_at_tstep(tstep, mask=fovmask)

    def sweep_duv_dt(self, uv_pair, extras=None):
        """Returns the mean local sweep speed of the instrument in the (u,v)
        directions.

        Input:
            uv_pair     a Pair of spatial indices (u,v).
            extras      an optional tuple or dictionary containing any extra
                        parameters required to define the timing of array
                        elements.

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """

        uv_pair = Pair.as_pair(uv_pair)
        tstep = uv_pair.as_scalar(self.cross_slit_uv_index)

        return self.duv_dt_basis / self.cadence.tstride_at_tstep(tstep)

    def time_shift(self, dtime):
        """Returns a copy of the observation object in which times have been
        shifted by a constant value.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = RasterSlit(self.axes, self.uv_size,
                         self.cadence.time_shift(dtime),
                         self.fov, self.path_id, self.frame_id)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops_.cadence.metronome import Metronome
from oops_.cadence.dual import DualCadence

class Test_RasterSlit(unittest.TestCase):

    def runTest(self):

        from oops_.fov.flat import Flat

        fov = Flat((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=("ufast","vslow"), det_size=1,
                         cadence=cadence, fov=fov,
                         path_id="SSB", frame_id="J2000")

        indices = Tuple([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        # uvt() with fovmask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(time.mask)
        self.assertEqual(time, slow_cadence.tstride * indices.as_scalar(1) +
                               fast_cadence.tstride * indices.as_scalar(0))
        self.assertEqual(uv.as_scalar(0), indices.as_scalar(0))
        self.assertEqual(uv.as_scalar(1), 0.5)

        # uvt() with fovmask == True
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6],
                         (slow_cadence.tstride * indices.as_scalar(1) +
                          fast_cadence.tstride * indices.as_scalar(0))[:6])
        self.assertEqual(uv[:6].as_scalar(0), indices[:6].as_scalar(0))
        self.assertEqual(uv[:6].as_scalar(1), 0.5)

        # uvt_range() with fovmask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min.as_scalar(0), indices.as_scalar(0))
        self.assertEqual(uv_min.as_scalar(1), 0)
        self.assertEqual(uv_max.as_scalar(0), indices.as_scalar(0) + 1)
        self.assertEqual(uv_max.as_scalar(1), 1)
        self.assertEqual(time_min,
                         slow_cadence.tstride * indices.as_scalar(1) +
                         fast_cadence.tstride * indices.as_scalar(0))
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        # uvt_range() with fovmask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min.as_scalar(0), indices.as_scalar(0))
        self.assertEqual(uv_min.as_scalar(1), 0)
        self.assertEqual(uv_max.as_scalar(0), indices.as_scalar(0) + 1)
        self.assertEqual(uv_max.as_scalar(1), 1)
        self.assertEqual(time_min, slow_cadence.tstride * indices.as_scalar(1) +
                                   fast_cadence.tstride * indices.as_scalar(0))
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        # uvt_range() with fovmask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.as_scalar(0)[:2], indices.as_scalar(0)[:2])
        self.assertEqual(uv_min.as_scalar(1)[:2], 0)
        self.assertEqual(uv_max.as_scalar(0)[:2], indices.as_scalar(0)[:2] + 1)
        self.assertEqual(uv_max.as_scalar(1)[:2], 1)
        self.assertEqual(time_min[:2],
                         (slow_cadence.tstride * indices.as_scalar(1) +
                          fast_cadence.tstride * indices.as_scalar(0))[:2])
        self.assertEqual(time_max[:2], time_min[:2] + fast_cadence.texp)

        # times_at_uv() with fovmask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, slow_cadence.tstride * uv.as_scalar(1) +
                                fast_cadence.tstride * uv.as_scalar(0))
        self.assertEqual(time1, time0 + fast_cadence.texp)

        # times_at_uv() with fovmask == True
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4],
                         (slow_cadence.tstride * uv.as_scalar(1) +
                          fast_cadence.tstride * uv.as_scalar(0))[:4])
        self.assertEqual(time1[:4], time0[:4] + fast_cadence.texp)

        ####################################
        # Alternative axis order ("uslow","vfast")

        fov = Flat((0.001,0.001), (1,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        fast_cadence = Metronome(tstart=0., tstride=0.5, texp=0.5, steps=20)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=("uslow","vfast"), det_size=1,
                         cadence=cadence, fov=fov,
                         path_id="SSB", frame_id="J2000")

        indices = Tuple([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.as_scalar(0), 0.5)
        self.assertEqual(uv.as_scalar(1), indices.as_scalar(1))
        self.assertEqual(time, slow_cadence.tstride * indices.as_scalar(0) +
                               fast_cadence.tstride * indices.as_scalar(1))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.as_scalar(0), 0)
        self.assertEqual(uv_min.as_scalar(1), indices.as_scalar(1))
        self.assertEqual(uv_max.as_scalar(0), 1)
        self.assertEqual(uv_max.as_scalar(1), indices.as_scalar(1) + 1)
        self.assertEqual(time_min, slow_cadence.tstride * indices.as_scalar(0) +
                                   fast_cadence.tstride * indices.as_scalar(1))
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        (time0,time1) = obs.times_at_uv(indices)

        self.assertEqual(time0, slow_cadence.tstride * indices.as_scalar(0) +
                                fast_cadence.tstride * indices.as_scalar(1))
        self.assertEqual(time1, time0 + fast_cadence.texp)

        ####################################
        # Alternative det_size and texp for discontinuous indices

        fov = Flat((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=8., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.5, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=("ufast","vslow"), det_size=0.5,
                         cadence=cadence, fov=fov,
                         path_id="SSB", frame_id="J2000")

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
        fov = Flat((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=11., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=("ufast","vslow"), det_size=0.5,
                         cadence=cadence, fov=fov,
                         path_id="SSB", frame_id="J2000")

        self.assertEqual(obs.time[1], 218.8)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1],  5.)
        self.assertEqual(obs.uvt((5,5))[1], 60.)
        self.assertEqual(obs.uvt((5.0, 5.5))[1], 60.)
        self.assertEqual(obs.uvt((5.5, 5.0))[1], 60.4)
        self.assertEqual(obs.uvt((5.5, 5.5))[1], 60.4)

        eps = 1.e-15
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

        # Alternative with uv_size and texp and axes
        fov = Flat((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=("a","vslow","b","ufast","c"), det_size=0.5,
                         cadence=cadence, fov=fov,
                         path_id="SSB", frame_id="J2000")

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
