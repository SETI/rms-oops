################################################################################
# oops_/obs/pixel.py: Subclass Pixel of class Observation
#
# 7/24/12 MRS - Created.
################################################################################

import numpy as np

from oops_.array.all import *
from oops_.obs.observation_ import Observation
from oops_.event import Event

class Pixel(Observation):
    """A Pixel is a subclass of Observation consisting of one or more
    measurements obtained from a single rectangular pixel.

    Generalization to other pixel shapes is TDB. 7/24/12 MRS
    """

    def __init__(self, axes, cadence, fov, path_id, frame_id, **subfields):
        """Constructor for a Pixel observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of "t" should
                        appear at the location of the array's time-axis.

            cadence     a Cadence object defining the start time and duration of
                        each consecutive measurement.
            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a Pixel object, both axes of the
                        FOV must have length 1.
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
        self.u_axis = -1
        self.v_axis = -1
        self.t_axis = self.axes.index("t")

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime
        self.scalar_times = (Scalar(self.time[0]), Scalar(self.time[1]))

        assert self.fov.uv_shape == (1,1)
        self.uv_shape = [1,1]

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
        tstep = indices.as_scalar(self.t_axis)

        time = self.cadence.time_at_tstep(tstep)
        uv = Pair((0.5,0.5))

        if fovmask:
            is_inside = self.cadence.time_is_inside(time, inclusive=True)
            if not np.all(is_inside):
                mask = indices.mask | ~is_inside
                time.mask = mask

                uv_vals = np.empty(indices.shape + [2])
                uv_vals[...] = 0.5
                uv = Pair(uv_vals, mask)

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
        tstep = indices.as_scalar(self.t_axis)

        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep,
                                                                mask=fovmask)
        uv_min = Pair.ZERO
        uv_max = Pair.ONES

        if fovmask:
            mask = time_min.mask
            if np.any(mask):
                uv_min_vals = np.zeros(indices.shape + [2])
                uv_max_vals = np.ones(indices.shape  + [2])

                uv_min = Pair(uv_min_vals, mask)
                uv_max = Pair(uv_max_vals, mask)

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

        if not fovmask: return self.scalar_times

        uv_pair = Pair.as_pair(uv_pair)
        is_inside = self.uv_is_inside(uv_pair, inclusive=True)
        if not np.all(is_inside):
            mask = uv_pair.mask | ~is_inside

            time0_vals = np.empty(uv_pair.shape)
            time1_vals = np.empty(uv_pair.shape)

            time0_vals[...] = self.time[0]
            time1_vals[...] = self.time[1]

            time0 = Scalar(time0_vals, mask)
            time1 = Scalar(time1_vals, mask)

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

        obs = Pixel(self.axes, self.cadence.time_shift(dtime),
                    self.fov, self.path_id, self.frame_id)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

    ############################################################################
    # Overrides of Observation class methods
    ############################################################################

    def event_at_grid(self, meshgrid, time=None):
        """Returns an event object describing the arrival of a photon at a set
        of locations defined by the given meshgrid. This version overrides the
        default definition to apply the timing for each pixel of a time-sequence
        by default.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view.
            time        a Scalar of times; None to use the midtime of each
                        pixel in the meshgrid.

        Return:         the corresponding event.
        """

        assert len(self.cadence.shape) == 1

        if time is None:
            tstep = np.arange(self.cadence.shape[0]) + 0.5
            time = self.cadence.time_at_tstep(tstep)
            time = time.append_axes(len(meshgrid.shape))

        event = Event(time, Vector3.ZERO, Vector3.ZERO,
                            self.path_id, self.frame_id)

        # Insert the arrival directions
        event.insert_subfield("arr", -meshgrid.los)

        return event

    def gridless_event(self, meshgrid, time=None):
        """Returns an event object describing the arrival of a photon at a set
        of locations defined by the given meshgrid. This version overrides the
        default definition to apply the timing for each pixel of a time-sequence
        by default.

        Input:
            meshgrid    a Meshgrid object describing the sampling of the field
                        of view.
            time        a Scalar of times; None to use the midtime of each
                        pixel in the meshgrid.

        Return:         the corresponding event.
        """

        assert len(self.cadence.shape) == 1

        if time is None:
            tstep = np.arange(self.cadence.shape[0]) + 0.5
            time = self.cadence.time_at_tstep(tstep)
            time = time.append_axes(len(meshgrid.shape))

        event = Event(time, Vector3.ZERO, Vector3.ZERO,
                            self.path_id, self.frame_id)

        return event

################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops_.cadence.metronome import Metronome

class Test_Pixel(unittest.TestCase):

    def runTest(self):

        from oops_.fov.flat import Flat

        fov = Flat((0.001,0.001), (1,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pixel(axes=("t"),
                    cadence=cadence, fov=fov, path_id="SSB", frame_id="J2000")

        indices = Tuple([(0,),(1,),(20,),(21,)])

        # uvt() with fovmask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(time.mask)
        self.assertEqual(time, cadence.tstride * indices.as_scalar())
        self.assertEqual(uv, (0.5,0.5))

        # uvt() with fovmask == True
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:3], cadence.tstride * indices.as_scalar()[:3])
        self.assertEqual(uv[:3], (0.5,0.5))

        # uvt_range() with fovmask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.as_scalar())
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with fovmask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices + (0.2,))

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.as_scalar())
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with fovmask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices + (0.2,),
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], (0,0))
        self.assertEqual(uv_max[:2], (1,1))
        self.assertEqual(time_min[:2], cadence.tstride *
                                       indices.as_scalar()[:2])
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        # times_at_uv() with fovmask == False
        uv = Pair([(0,0),(0,1),(1,0),(1,1),(1,2)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, obs.time[0])
        self.assertEqual(time1, obs.time[1])

        # times_at_uv() with fovmask == True
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:4], obs.time[0])
        self.assertEqual(time1[:4], obs.time[1])

        ####################################
        # Alternative axis order ("a","t")

        fov = Flat((0.001,0.001), (1,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pixel(axes=("a","t"),
                    cadence=cadence, fov=fov, path_id="SSB", frame_id="J2000")

        indices = Tuple([(0,0),(1,1),(0,20,),(1,21)])

        # uvt() with fovmask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(time.mask)
        self.assertEqual(time, cadence.tstride * indices.as_scalar(1))
        self.assertEqual(uv, (0.5,0.5))

        # uvt() with fovmask == True
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:3], cadence.tstride * indices[:3].as_scalar(1))
        self.assertEqual(uv[:3], (0.5,0.5))

        # uvt_range() with fovmask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.as_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with fovmask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.as_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with fovmask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices + (0.2,),
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:3], (0,0))
        self.assertEqual(uv_max[:3], (1,1))
        self.assertEqual(time_min[:3], cadence.tstride *
                                       indices.as_scalar(1)[:3])
        self.assertEqual(time_max[:3], time_min[:3] + cadence.texp)

        # times_at_uv() with fovmask == False
        uv = Pair([(0,0),(0,1),(1,0),(1,1),(1,2)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, obs.time[0])
        self.assertEqual(time1, obs.time[1])

        # times_at_uv() with fovmask == True
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:4], obs.time[0])
        self.assertEqual(time1[:4], obs.time[1])

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
