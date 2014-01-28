################################################################################
# oops_/obs/slit.py: Subclass Slit of class Observation
#
# 6/13/12 MRS - Created.
################################################################################

import numpy as np

from oops_.obs.observation_ import Observation
from oops_.array.all import *

class Slit(Observation):
    """A Slit is subclass of Observation consisting of a 2-D image constructed
    by rotating an instrument that has a 1-D array of sensors. This differs from
    a Pushbroom in that the FOV is intrinsically 1-D and static in the ICS. The
    secondary axis of the pixel array is obtained by sampling the slit at
    uniformly space intervals in time.
    """

    def __init__(self, axes, det_size,
                       cadence, fov, path_id, frame_id, **subfields):
        """Constructor for a Slit observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of "u" or "ut"
                        should appear at the location of the array's u-axis;
                        "vt" or "v" should appear at the location of the array's
                        v-axis. The "t" suffix is used for the one of these axes
                        that is emulated by time-sampling perpendicular to the
                        slit.
            det_size    the size of the detectors in FOV units parallel to the
                        slit. It will be < 1 if there are gaps between the
                        detectors.

            cadence     a Cadence object defining the start time and duration of
                        each consecutive measurement.
            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a Slit object, one of the axes of
                        the FOV must have length 1.
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

        self.axes = list(axes)
        assert (("u" in self.axes and "vt" in self.axes) or
                ("v" in self.axes and "ut" in self.axes))

        if "ut" in self.axes:
            self.u_axis = self.axes.index("ut")
            self.v_axis = self.axes.index("v")
            self.t_axis = self.u_axis
            self.along_slit_index = self.v_axis
            self.cross_slit_uv_index = 0
            self.along_slit_uv_index = 1
            self.uv_shape = [self.cadence.shape[0],
                             self.fov.uv_shape.vals[self.along_slit_index]]
        else:
            self.u_axis = self.axes.index("u")
            self.v_axis = self.axes.index("vt")
            self.t_axis = self.v_axis
            self.along_slit_index = self.u_axis
            self.cross_slit_uv_index = 1
            self.along_slit_uv_index = 0
            self.uv_shape = [self.fov.uv_shape.vals[self.along_slit_index],
                             self.cadence.shape[0]]

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        assert len(self.cadence.shape) == 1
        assert self.fov.uv_shape.vals[self.cross_slit_uv_index] == 1

        self.det_size = det_size
        self.slit_is_discontinuous = (self.det_size < 1)

        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

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

        slit_coord = indices.as_scalar(self.along_slit_index)
        if self.slit_is_discontinuous:
            slit_int = slit_coord.int()
            slit_coord = slit_int + (slit_coord - slit_int) * self.det_size

        uv_vals = np.empty(indices.shape + [2])
        uv_vals[..., self.along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_index] = 0.5
        uv = Pair(uv_vals, indices.mask)

        tstep = indices.as_scalar(self.t_axis)
        time = self.cadence.time_at_tstep(tstep)

        if fovmask:
            u_index = indices.vals[..., self.u_axis]
            v_index = indices.vals[..., self.v_axis]
            is_inside = ((u_index >= 0) &
                         (v_index >= 0) &
                         (u_index <= self.uv_shape[0]) &
                         (v_index <= self.uv_shape[1]))
            if not np.all(is_inside):
                mask = indices.mask | np.logical_not(is_inside)
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

        slit_coord = indices.as_scalar(self.along_slit_index)

        uv_vals = np.empty(indices.shape + [2], dtype="int")
        uv_vals[..., self.along_slit_uv_index] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_index] = 0
        uv_min = Pair(uv_vals, indices.mask)
        uv_max = uv_min + Pair.ONES

        tstep = indices.as_scalar(self.t_axis)
        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep)

        if fovmask:
            u_index = indices.vals[..., self.u_axis]
            v_index = indices.vals[..., self.v_axis]
            is_inside = ((u_index >= 0) &
                         (v_index >= 0) &
                         (u_index < self.uv_shape[0]) &
                         (v_index < self.uv_shape[1]))
            if not np.all(is_inside):
                mask = indices.mask | np.logical_not(is_inside)
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

        uv_pair = Pair.as_int(uv_pair)
        tstep = uv_pair.as_scalar(self.cross_slit_uv_index)
        (time0, time1) = self.cadence.time_range_at_tstep(tstep)

        if fovmask:
            u_index = uv_pair.vals[..., 0]
            v_index = uv_pair.vals[..., 1]
            is_inside = ((u_index >= 0) &
                         (v_index >= 0) &
                         (u_index <= self.uv_shape[0]) &
                         (v_index <= self.uv_shape[1]))
            if not np.all(is_inside):
                mask = uv_pair.mask | np.logical_not(is_inside)
                time0.mask = mask
                time1.mask = mask

        return (time0, time1)

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

        return Pair.ZERO

    def time_shift(self, dtime):
        """Returns a copy of the observation object in which times have been
        shifted by a constant value.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = Slit(self.axes, self.uv_size,
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

class Test_Slit(unittest.TestCase):

    def runTest(self):

        from oops_.fov.flat import Flat

        fov = Flat((0.001,0.001), (10,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Slit(axes=("u","vt"), det_size=1,
                   cadence=cadence, fov=fov, path_id="SSB", frame_id="J2000")

        indices = Tuple([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        # uvt() with fovmask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(time.mask)
        self.assertEqual(time, cadence.tstride * indices.as_scalar(1))
        self.assertEqual(uv.as_scalar(0), indices.as_scalar(0))
        self.assertEqual(uv.as_scalar(1), 0.5)

        # uvt() with fovmask == True
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], cadence.tstride * indices.as_scalar(1)[:6])
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
        self.assertEqual(time_min, cadence.tstride * indices.as_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

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
        self.assertEqual(time_min, cadence.tstride * indices.as_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

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
        self.assertEqual(time_min[:2], cadence.tstride *
                                       indices.as_scalar(1)[:2])
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        # times_at_uv() with fovmask == False
        uv = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, cadence.tstride * uv.as_scalar(1))
        self.assertEqual(time1, time0 + cadence.texp)

        # times_at_uv() with fovmask == True
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 6*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:6], cadence.tstride * uv.as_scalar(1)[:6])
        self.assertEqual(time1[:6], time0[:6] + cadence.texp)

        ####################################
        # Alternative axis order ("ut","v")

        fov = Flat((0.001,0.001), (1,20))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = Slit(axes=("ut","v"), det_size=1,
                   cadence=cadence, fov=fov, path_id="SSB", frame_id="J2000")

        indices = Tuple([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.as_scalar(0), 0.5)
        self.assertEqual(uv.as_scalar(1), indices.as_scalar(1))
        self.assertEqual(time, cadence.tstride * indices.as_scalar(0))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.as_scalar(0), 0)
        self.assertEqual(uv_min.as_scalar(1), indices.as_scalar(1))
        self.assertEqual(uv_max.as_scalar(0), 1)
        self.assertEqual(uv_max.as_scalar(1), indices.as_scalar(1) + 1)
        self.assertEqual(time_min, cadence.tstride * indices.as_scalar(0))
        self.assertEqual(time_max, time_min + cadence.texp)

        ####################################
        # Alternative det_size and texp for discontinuous indices

        fov = Flat((0.001,0.001), (1,20))
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Slit(axes=("ut","v"), det_size=0.8,
                   cadence=cadence, fov=fov, path_id="SSB", frame_id="J2000")

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1], 50.)
        self.assertEqual(obs.uvt((5,5))[1], 50.)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6      ,0))[1] - 60.) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 62.) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 64.) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 66.) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 68.) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 70.) < delta)

        self.assertEqual(obs.uvt((0,0))[0], (0.5,0.))
        self.assertEqual(obs.uvt((5,0))[0], (0.5,0.))
        self.assertEqual(obs.uvt((5,5))[0], (0.5,5.))

        self.assertTrue(abs(obs.uvt((6      ,0))[0] - (0.5,0.)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (0.5,1.)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (0.5,2.)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (0.5,3.)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (0.5,4.)) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,5))[0] - (0.5,5.)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (0.5,6.)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ))[0] - (0.5,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((2, 1.25   ))[0] - (0.5,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((3, 2.5    ))[0] - (0.5,2.4)) < delta)
        self.assertTrue(abs(obs.uvt((4, 3.75   ))[0] - (0.5,3.6)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5 - eps))[0] - (0.5,4.8)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5.     ))[0] - (0.5,5.0)) < delta)

        # Alternative with uv_size and texp and axes
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Slit(axes=("a","v","b","ut","c"), det_size=0.8,
                   cadence=cadence, fov=fov, path_id="SSB", frame_id="J2000")

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((1,0,3,0,4))[1],  0.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1], 50.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1], 50.)

        eps = 1.e-15
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((1,0,0,6      ,0))[1] - 60.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.25   ,0))[1] - 62.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.5    ,0))[1] - 64.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.75   ,0))[1] - 66.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7 - eps,0))[1] - 68.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7.     ,0))[1] - 70.) < delta)

        self.assertEqual(obs.uvt((0,0,0,0,0))[0], (0.5,0.))
        self.assertEqual(obs.uvt((0,0,0,5,0))[0], (0.5,0.))
        self.assertEqual(obs.uvt((0,5,0,5,0))[0], (0.5,5.))

        self.assertTrue(abs(obs.uvt((1,0,4,6      ,7))[0] - (0.5,0.)) < delta)
        self.assertTrue(abs(obs.uvt((1,1,4,6.2    ,7))[0] - (0.5,1.)) < delta)
        self.assertTrue(abs(obs.uvt((1,2,4,6.4    ,7))[0] - (0.5,2.)) < delta)
        self.assertTrue(abs(obs.uvt((1,3,4,6.6    ,7))[0] - (0.5,3.)) < delta)
        self.assertTrue(abs(obs.uvt((1,4,4,6.8    ,7))[0] - (0.5,4.)) < delta)
        self.assertTrue(abs(obs.uvt((1,5,4,7 - eps,7))[0] - (0.5,5.)) < delta)
        self.assertTrue(abs(obs.uvt((1,6,4,7.     ,7))[0] - (0.5,6.)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ,4,1,7))[0] - (0.5,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((1, 1.25   ,4,2,7))[0] - (0.5,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (0.5,2.4)) < delta)
        self.assertTrue(abs(obs.uvt((1, 3.75   ,4,4,7))[0] - (0.5,3.6)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5 - eps,4,5,7))[0] - (0.5,4.8)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5.     ,4,5,7))[0] - (0.5,5.0)) < delta)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
