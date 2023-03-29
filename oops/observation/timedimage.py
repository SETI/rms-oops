################################################################################
# oops/observation/timedimage.py: Subclass TimedImage of class Observation
################################################################################

import numpy as np
from polymath import Pair, Vector, Qube, Boolean, Scalar

from oops.observation           import Observation
from oops.observation.snapshot  import Snapshot
from oops.frame                 import Frame
from oops.path                  import Path

class TimedImage(Observation):
    """An image in which the individual pixels have distinct timing.

    Pixel timing is defined by a 1-D or 2-D cadence.
    """

    # NOTE:
    # This class now encompasses the earlier subclasses Pushbroom, Pushframe,
    # RasterScan, RasterSlit, and Slit. (In other words, every observation
    # subclass with two spatial dimensions except Snapshot.)

    INVENTORY_IMPLEMENTED = True

    #===========================================================================
    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        """Constructor for a Pushframe.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. One of these strings must
                        begin with "u", and the other must begin with "v", to
                        indicate the locations of the spatial axes. If the image
                        has a 1-D cadence, then "t" should be appended to the
                        name of the axis containing time dependence. If both
                        axes have time dependence, one should have the suffix
                        "fast" and the other should have suffix "slow".

            cadence     a 1-D or 2-D Cadence object defining the start and stop
                        time of each pixel.

            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion and possible
                        time-dependence. It maps between spatial coordinates
                        (u,v) and instrument coordinates (x,y).

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

        # Static FOV
        self.fov = fov
        self.fov_shape = tuple(self.fov.uv_shape.vals)
        self._has_unit_fov_axis = (self.fov_shape[0] == 1 or
                                   self.fov_shape[1] == 1)

        # Axes
        self.axes = tuple(axes)

        u_axes = [k for k in range(len(self.axes))
                  if self.axes[k].startswith('u')]
        v_axes = [k for k in range(len(self.axes))
                  if self.axes[k].startswith('v')]
        if len(u_axes) != 1 or len(v_axes) != 1:
            raise ValueError('invalid axis labels for TimedImage: %s'
                             % str(self.axes))

        self.u_axis = u_axes[0]
        self.v_axis = v_axes[0]

        u_suffix = self.axes[self.u_axis][1:]
        v_suffix = self.axes[self.v_axis][1:]

        if u_suffix == 't' and v_suffix == '':
            self.t_axis = self.u_axis
            self._t_uv_axis = 0
        elif u_suffix == '' and v_suffix == 't':
            self.t_axis = self.v_axis
            self._t_uv_axis = 1
        elif u_suffix == 'fast' and v_suffix == 'slow':
            self.t_axis = (self.v_axis, self.u_axis)
            self._fast_t_uv_axis = 0
        elif u_suffix == 'slow' and v_suffix == 'fast':
            self.t_axis = (self.u_axis, self.v_axis)
            self._fast_t_uv_axis = 1
        else:
            raise ValueError('invalid axis labels for TimedImage: "%s", "%s"'
                             % (self.axes[self.u_axis], self.axes[self.v_axis]))

        self.swap_uv = (self.u_axis > self.v_axis)
        self._time_is_1d = not isinstance(self.t_axis, tuple)

        # Cadence
        self.cadence = cadence
        if self._time_is_1d:
            if len(self.cadence.shape) != 1:
                raise ValueError('TimedImage axes requires 1-D cadence')
        else:
            if len(self.cadence.shape) != 2:
                raise ValueError('TimedImage axes requires 2-D cadence')

        # Timing
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        # Shape / Size
        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.fov_shape[0]
        self.shape[self.v_axis] = self.fov_shape[1]

        # This is the (u,v) shape of the observation, not necessarily that of
        # the FOV.
        self.uv_shape = [self.fov_shape[0], self.fov_shape[1]]

        # Cadence overrides the shape as defined by the FOV
        # However, the inventory method will require serious modification for
        # observations in which the cadence defines one dimension, not the FOV.
        if self._time_is_1d:
            t_size = self.cadence.shape[0]
            if t_size < self.shape[self.t_axis]:
                raise ValueError('TimedImage FOV and cadence have incompatible '
                                 + 'shapes')
            self._extended_fov = (t_size > self.shape[self.t_axis])
            self.shape[self.t_axis] = t_size
            self.uv_shape[self._t_uv_axis] = t_size
        else:
            if self.shape[self.t_axis[0]] not in (self.cadence.shape[0], 1):
                raise ValueError('TimedImage FOV and cadence have incompatible '
                                 + 'shapes')
            t_size = self.cadence.shape[1]
            if t_size < self.shape[self.t_axis[1]]:
                raise ValueError('TimedImage FOV and cadence have incompatible '
                                 + 'shapes')
            self._extended_fov = (t_size > self.shape[self.t_axis[1]])
            self.shape[self.t_axis[1]] = t_size
            self.uv_shape[self._fast_t_uv_axis] = t_size

        self.INVENTORY_IMPLEMENTED = not self._extended_fov

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        # TODO: implement inventory and related methods for an extended FOV.

        if self._extended_fov:
            self.snapshot = None
        else:
            snapshot_axes = list(self.axes)     # a copy
            snapshot_axes[self.u_axis] = 'u'
            snapshot_axes[self.v_axis] = 'v'
            snapshot_tstart = self.cadence.time[0]
            snapshot_texp = self.cadence.time[1] - self.cadence.time[0]

            self.snapshot = Snapshot(snapshot_axes, snapshot_tstart,
                                     snapshot_texp, self.fov,
                                     self.path, self.frame, **subfields)

    def __getstate__(self):
        return (self.axes, self.cadence, self.fov, self.path, self.frame,
                self.subfields)

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

        # If an FOV axis has unit length, we always land at 0.5
        if self._has_unit_fov_axis:
            if uv.is_float():
                uv = uv.copy()
            else:
                uv = uv.as_float()

            if self.fov_shape[0] == 1:
                uv.vals[..., 0] = 0.5
            if self.fov_shape[1] == 1:
                uv.vals[..., 1] = 0.5

        new_mask = False
        if self._time_is_1d:
            tstep = indices.to_scalar(self.t_axis)

            # Re-mask the time-independent axis if necessary
            if remask:
                not_t_vals = uv.vals[..., 1 - self._t_uv_axis]
                not_t_max = self.uv_shape[1 - self._t_uv_axis]
                new_mask = (not_t_vals < 0) | (not_t_vals > not_t_max)
                if not np.any(new_mask):
                    new_mask = False
        else:
            tstep = indices.to_pair(self.t_axis)

        time = self.cadence.time_at_tstep(tstep, remask=remask, derivs=derivs)

        # Merge masks if necessary
        if remask:
            new_mask = Qube.or_(new_mask, time.mask)
            uv = uv.remask(new_mask)
            time = time.remask(new_mask)

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

        # If an FOV axis has unit length, we always land at range (0,1)
        if self._has_unit_fov_axis:
            uv_min = uv_min.copy()
            if self.fov_shape[0] == 1:
                uv_min.vals[..., 0] = 0
            if self.fov_shape[1] == 1:
                uv_min.vals[..., 1] = 0

        new_mask = False
        if self._time_is_1d:
            tstep = indices.to_scalar(self.t_axis)

            # Re-mask the time-independent axis if necessary
            if remask:
                not_t_vals = uv.vals[..., 1 - self._t_uv_axis]
                not_t_max = self.uv_shape[1 - self._t_uv_axis]
                new_mask = (not_t_vals < 0) | (not_t_vals > not_t_max)
        else:
            tstep = indices.to_pair(self.t_axis)

        (time_min,
         time_max) = self.cadence.time_range_at_tstep(tstep, remask=remask)

        # Merge masks if necessary
        if remask:
            if np.any(new_mask):
                new_mask = Qube.or_(new_mask, time_min.mask)
                uv_min = uv_min.remask(new_mask)
                time_min = time_min.remask(new_mask)
                time_max = time_max.remask(new_mask)
            else:
                uv_min = uv_min.remask(time_min.mask)

        return (uv_min, uv_min + Pair.INT11, time_min, time_max)

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

        if self._time_is_1d:
            return self.time_range_at_uv_1d(uv_pair, axis=self._t_uv_axis,
                                                     remask=remask)
        else:
            return self.time_range_at_uv_2d(uv_pair, fast=self._fast_t_uv_axis,
                                                     remask=remask)

    #===========================================================================
    def uv_range_at_time(self, time, remask=False):
        """The (u,v) range of spatial pixels observed at the specified time.

        Input:
            time        a Scalar of time values in seconds TDB.
            remask      True to mask values outside the time limits.

        Return:         (uv_min, uv_max)
            uv_min      the lower (u,v) corner Pair of the area observed at the
                        specified time.
            uv_max      the upper (u,v) corner Pair of the area observed at the
                        specified time.
        """

        if self._time_is_1d:
            return self.uv_range_at_time_1d(time, shape=self.uv_shape,
                                                  axis=self._t_uv_axis,
                                                  remask=remask)
        else:
            return self.uv_range_at_time_2d(time, shape=self.uv_shape,
                                                  slow=(1-self._fast_t_uv_axis),
                                                  fast=self._fast_t_uv_axis,
                                                  remask=remask)

    #===========================================================================
    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        return TimedImage(self.axes, self.cadence.time_shift(dtime),
                          self.fov, self.path, self.frame, **self.subfields)

    #===========================================================================
    def inventory(self, *args, **kwargs):
        """Info about the bodies that appear unobscured inside the FOV. See
        Snapshot.inventory() for details.

        WARNING: Not properly updated for class PushFrame. Use at your own risk.
        This operates by returning every body that would have been inside the
        FOV of this observation if it were instead a Snapshot, evaluated at the
        given tfrac.
        """

        # TODO
        if self._extended_fov:
            raise NotImplementedError('inventory is not implemented for '
                                      'TimedImage with cadence-extended FOV')

        return self.snapshot.inventory(*args, **kwargs)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_TimedImage(unittest.TestCase):

    def runTest(self):

        from oops.cadence.metronome   import Metronome
        from oops.cadence.dualcadence import DualCadence
        from oops.cadence.tdicadence  import TDICadence
        from oops.fov.flatfov         import FlatFOV

        ########################################################################
        # Old RasterScan unit tests
        ########################################################################

        RasterScan = TimedImage

        ####################################################
        # Continuous observation, shape (10,20)
        # Axes are (fast,slow)
        ####################################################

        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                         path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(0,21),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

        # uvt() with remask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, [0, 100, 190, 10, 110, 200, 190, 200])
        self.assertEqual(uv, Pair.as_pair(indices))

        # uvt() with remask == True
        (uv, time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + 2*[True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], [0, 100, 190, 10, 110, 200])
        self.assertEqual(uv[:6], Pair.as_pair(indices)[:6])

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices_))
        self.assertEqual(uv_max, Pair.as_pair(indices_) + (1,1))
        self.assertEqual(time_min, [0, 100, 190,  9, 109, 199, 190, 199])
        self.assertEqual(time_max, [1, 101, 191, 10, 110, 200, 191, 200])

        # uvt_range() with remask == True
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices, remask=True)

        self.assertEqual(Boolean(uv_min.mask), 6*[False] + 2*[True])
        self.assertEqual(Boolean(uv_max.mask), 6*[False] + 2*[True])
        self.assertEqual(Boolean(time_min.mask), 6*[False] + 2*[True])
        self.assertEqual(Boolean(time_max.mask), 6*[False] + 2*[True])

        self.assertEqual(uv_min[:6], Pair.as_pair(indices_)[:6])
        self.assertEqual(uv_max[:6], Pair.as_pair(indices_)[:6] + (1,1))
        self.assertEqual(time_min[:6], [0, 100, 190,  9, 109, 199])
        self.assertEqual(time_max[:6], time_min[:6] + fast_cadence.texp)

        # uvt() with remask == False, non-integer indices
        non_ints = indices + (0.2, 0.9)
        (uv, time) = obs.uvt(non_ints)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.time_at_tstep(uv.swapxy()))
        self.assertEqual(uv, Pair.as_pair(non_ints))

        # uvt() with remask == True, non-integer indices
        non_ints = indices + (0.2, 0.9)
        (uv, time) = obs.uvt(non_ints, remask=True)

        self.assertEqual(Boolean(uv.mask), 2*[False] + 6*[True])
        self.assertEqual(Boolean(time.mask), 2*[False] + 6*[True])
        self.assertEqual(time[:2],
                         (slow_cadence.tstride * non_ints.to_scalar(1).int() +
                          fast_cadence.tstride * non_ints.to_scalar(0))[:2])
        self.assertEqual(uv[:2], Pair.as_pair(non_ints)[:2])

        # uvt_range() with remask == False, non-integer indices
        non_ints = indices + (0.2, 0.9)
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min, cadence.time_range_at_tstep(Pair.as_pair(non_ints).swapxy())[0])
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        # uvt_range() with remask == True, non-integer indices
        non_ints = indices + (0.2, 0.9)
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints, remask=True)

        self.assertEqual(Boolean(uv_min.mask), 2*[False] + 6*[True])
        self.assertEqual(Boolean(uv_max.mask), 2*[False] + 6*[True])
        self.assertEqual(Boolean(time_min.mask), 2*[False] + 6*[True])
        self.assertEqual(Boolean(time_max.mask), 2*[False] + 6*[True])

        self.assertEqual(uv_min[:2], Pair.as_pair(indices)[:2])
        self.assertEqual(uv_max[:2], Pair.as_pair(indices)[:2] + (1,1))
        self.assertEqual(time_min[:2],
                         (slow_cadence.tstride * non_ints.to_scalar(1).int() +
                          fast_cadence.tstride * non_ints.to_scalar(0).int())[:2])
        self.assertEqual(time_max[:2], time_min[:2] + fast_cadence.texp)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

        (time0, time1) = obs.time_range_at_uv(uv)

        self.assertEqual(time0, [0, 190, 9, 199, 199])
        self.assertEqual(time1, time0 + fast_cadence.texp)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], [0, 190, 9, 199])
        self.assertEqual(time1[:4], time0[:4] + fast_cadence.texp)

        ####################################################
        # Fast cadence is discontinuous
        # Axes are (slow,fast)
        # Shape (10,20)
        # [[0-1, 10-11, 20-21, ..., 190-191],
        #  [1000-1001, 1010-1011, ..., 1190-1191],
        #  ...
        #  [9000-9001, 9010-9011, ..., 9190, 9191]]
        ####################################################

        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=1000., texp=1., steps=10)
        fast_cadence = Metronome(tstart=0., tstride=10., texp=1., steps=20)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('uslow','vfast'), cadence=cadence, fov=fov,
                                                 path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, [0, 100, 191, 9000, 9100, 9191, 9191])
        self.assertEqual(uv, Pair.as_pair(indices))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min, Pair.as_pair(indices_))
        self.assertEqual(uv_max, Pair.as_pair(indices_) + (1,1))
        self.assertEqual(time_min, cadence.time_range_at_tstep(indices_)[0])
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        (time0,time1) = obs.time_range_at_uv(indices)

        self.assertEqual(time0, cadence.time_range_at_tstep(indices_)[0])
        self.assertEqual(time1, time0 + fast_cadence.texp)

        ####################################################
        # Fast cadence is discontinuous
        # Axes are (fast,slow)
        # Shape (10,20)
        ####################################################

        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('ufast','vslow'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

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
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (6.2,1.)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (6.4,2.)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (6.6,3.)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (6.8,4.)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (7.0,6.)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ))[0] - (1.,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((2, 1.2    ))[0] - (2.,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((3, 2.5    ))[0] - (3.,2.5)) < delta)
        self.assertTrue(abs(obs.uvt((4, 3.8    ))[0] - (4.,3.8)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5.     ))[0] - (5.,5.0)) < delta)

        ############################################################
        # Alternative tstride for even more discontinuous indices
        ############################################################

        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=11., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('ufast','vslow'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 218.8)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1],  5.)
        self.assertEqual(obs.uvt((5,5))[1], 60.)
        self.assertEqual(obs.uvt((5.0, 5.5))[1], 60.)
        self.assertEqual(obs.uvt((5.5, 5.0))[1], 60.4)
        self.assertEqual(obs.uvt((5.5, 5.5))[1], 60.4)

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue((obs.uvt((6.     ,0.))[1] - 6.0).abs() < delta)
        self.assertTrue((obs.uvt((6.25   ,0.))[1] - 6.2).abs() < delta)
        self.assertTrue((obs.uvt((6.5    ,0.))[1] - 6.4).abs() < delta)
        self.assertTrue((obs.uvt((6.75   ,0.))[1] - 6.6).abs() < delta)
        self.assertTrue((obs.uvt((7. -eps,0.))[1] - 6.8).abs() < delta)
        self.assertTrue((obs.uvt((7.     ,0.))[1] - 7.0).abs() < delta)

        self.assertTrue((obs.uvt((9.      ,0.))[1] -  9.0).abs() < delta)
        self.assertTrue((obs.uvt((9.25    ,0.))[1] -  9.2).abs() < delta)
        self.assertTrue((obs.uvt((9.5     ,0.))[1] -  9.4).abs() < delta)
        self.assertTrue((obs.uvt((9.75    ,0.))[1] -  9.6).abs() < delta)
        self.assertTrue((obs.uvt((10 - eps,0.))[1] -  9.8).abs() < delta)
        self.assertTrue((obs.uvt((0.      ,1.))[1] - 11.0).abs() < delta)

        self.assertTrue((obs.uvt((6.00, 0.    ))[1] -  6.0).abs() < delta)
        self.assertTrue((obs.uvt((6.25, 0.    ))[1] -  6.2).abs() < delta)
        self.assertTrue((obs.uvt((6.25, 1.    ))[1] - 17.2).abs() < delta)
        self.assertTrue((obs.uvt((6.25, 2.-eps))[1] - 17.2).abs() < delta)
        self.assertTrue((obs.uvt((6.25, 2.    ))[1] - 28.2).abs() < delta)

        # Test the upper edge
        pair = (10-eps, 20-eps)
        self.assertEqual(obs.uvt(pair)[0], pair)
        self.assertTrue((obs.uvt(pair)[1] - 218.8).abs() < delta)

        pair = (10, 20-eps)
        self.assertEqual(obs.uvt(pair)[0], pair)
        self.assertTrue((obs.uvt(pair)[1] - 218.8).abs() < delta)

        pair = (10-eps, 20)
        self.assertEqual(obs.uvt(pair)[0], pair)
        self.assertTrue((obs.uvt(pair)[1] - 218.8).abs() < delta)

        pair = (10, 20)
        self.assertEqual(obs.uvt(pair)[0], pair)
        self.assertTrue((obs.uvt(pair)[1] - 218.8).abs() < delta)

        self.assertTrue(obs.uvt((10+eps, 20), True)[0].mask)
        self.assertTrue(obs.uvt((10, 20+eps), True)[0].mask)

        # Try all at once
        indices = Pair([(10-eps,20-eps), (10,20-eps), (10-eps,20), (10,20),
                        (10+eps,20), (10,20+eps)])
        (test_uv, time) = obs.uvt(indices, remask=True)

        self.assertEqual(Boolean(test_uv.mask), 4*[False] + 2*[True])
        self.assertEqual(test_uv[:4], indices[:4])
        self.assertTrue(((time[:4] - 218.8).abs() < delta).all())
        self.assertEqual(Boolean(time.mask), test_uv.mask)

        ############################################################
        # Alternative texp and axes
        ############################################################

        fov = FlatFOV((0.001,0.001), (10,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterScan(axes=('a','vslow','b','ufast','c'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

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

        self.assertTrue(abs(obs.uvt((1,0,4,6   ,7))[0] - (6.0,0.)) < delta)
        self.assertTrue(abs(obs.uvt((1,1,4,6.2 ,7))[0] - (6.2,1.)) < delta)
        self.assertTrue(abs(obs.uvt((1,2,4,6.4 ,7))[0] - (6.4,2.)) < delta)
        self.assertTrue(abs(obs.uvt((1,3,4,6.6 ,7))[0] - (6.6,3.)) < delta)
        self.assertTrue(abs(obs.uvt((1,4,4,6.8 ,7))[0] - (6.8,4.)) < delta)
        self.assertTrue(abs(obs.uvt((1,6,4,7.  ,7))[0] - (7.0,6.)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ,4,1,7))[0] - (1.,0.0)) < delta)
        self.assertTrue(abs(obs.uvt((1, 1.2    ,4,2,7))[0] - (2.,1.2)) < delta)
        self.assertTrue(abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (3.,2.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 3.7    ,4,4,7))[0] - (4.,3.7)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5.     ,4,5,7))[0] - (5.,5.0)) < delta)

        ########################################################################
        # Old Pushbroom unit tests
        ########################################################################

        Pushbroom = TimedImage

        ########################################
        # Overall shape (10,20)
        # Time is second axis; time = v * 10.
        ########################################

        flatfov = FlatFOV((0.001,0.001), (10,20))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pushbroom(axes=('u','vt'), cadence=cadence, fov=flatfov,
                                         path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        tstep = indices.to_scalar(1)

        indices_ = indices.copy()   # clipped at top
        indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

        # uvt() with remask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.time_at_tstep(tstep))
        self.assertEqual(uv, Pair.as_pair(indices))

        # uvt() with remask == True
        (uv, time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], cadence.tstride * indices.to_scalar(1)[:6])
        self.assertEqual(uv[:6], Pair.as_pair(indices)[:6])

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices_))
        self.assertEqual(uv_max, Pair.as_pair(indices_) + (1,1))
        self.assertEqual(time_min, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with remask == False, new indices
        non_ints = indices + (0.2,0.9)
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with remask == True, new indices
        non_ints = indices + (0.2,0.9)
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints,
                                                             remask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], Pair.as_pair(indices)[:2])
        self.assertEqual(uv_max[:2], Pair.as_pair(indices)[:2] + (1,1))
        self.assertEqual(time_min[:2], cadence.time_range_at_tstep(tstep)[0][:2])
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])
        tstep = uv.to_scalar(1)

        uv_ = uv.copy()
        uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1
        uv_.vals[:,1][uv_.vals[:,1] == 20] -= 1

        (time0, time1) = obs.time_range_at_uv(uv)
        self.assertEqual(time0, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time1, time0 + cadence.texp)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)
        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], cadence.tstride * uv_.to_scalar(1)[:4])
        self.assertEqual(time1[:4], time0[:4] + cadence.texp)

        ########################################
        # Alternative axis order ('ut','v')
        # Overall shape (10,20)
        # Time is first axis; time = v * 10.
        ########################################

        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = Pushbroom(axes=('ut','v'), cadence=cadence, fov=flatfov,
                                         path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

        (uv, time) = obs.uvt(indices)

        uv_ = uv.copy()
        uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1
        uv_.vals[:,1][uv_.vals[:,1] == 20] -= 1

        self.assertEqual(uv, Pair.as_pair(indices))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(0))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min, Pair.as_pair(indices_))
        self.assertEqual(uv_max, Pair.as_pair(indices_) + (1,1))
        self.assertEqual(time_min, cadence.tstride * indices_.to_scalar(0))
        self.assertEqual(time_max, time_min + cadence.texp)

        (time0, time1) = obs.time_range_at_uv(indices)

        self.assertEqual(time0, cadence.tstride * uv_.to_scalar(0))
        self.assertEqual(time1, time0 + cadence.texp)

        ########################################################
        # Alternative texp for discontinuous time index
        # Overall shape (10,20)
        # Time is first axis; time = [0-8, 10-18, ..., 90-98]
        ########################################################

        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Pushbroom(axes=('ut','v'), cadence=cadence, fov=flatfov,
                                         path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1], 50.)
        self.assertEqual(obs.uvt((5,5))[1], 50.)

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6      ,0))[1] - 60.) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,0))[1] - 62.) < delta)
        self.assertTrue(abs(obs.uvt((6.5    ,0))[1] - 64.) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,0))[1] - 66.) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,0))[1] - 68.) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,0))[1] - 70.) < delta)

        self.assertEqual(obs.uvt((0,0))[0], (0.,0.))
        self.assertEqual(obs.uvt((5,0))[0], (5.,0.))
        self.assertEqual(obs.uvt((5,5))[0], (5.,5.))

        self.assertTrue(abs(obs.uvt((6      ,0))[0] - (6.0,0.)) < delta)
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (6.2,1.)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (6.4,2.)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (6.6,3.)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (6.8,4.)) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,5))[0] - (7.0,5.)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (7.0,6.)) < delta)

        # Test the upper edge
        uv_list = []
        uvt_list = []
        for i,u in enumerate([10.-eps, 10., 10.+eps]):
          for j,v in enumerate([20.-eps, 20., 20.+eps]):
            uv_list.append((u,v))

            uvt = obs.uvt((u,v), remask=True)
            uvt_list.append(uvt)
            if (i < 2) and (j < 2):
                self.assertEqual(uvt[0], (u,v))
            else:
                self.assertEqual(uvt[0], Pair.MASKED)

            if (i < 2) and (j < 2):
                self.assertTrue((uvt[1] - (10. * u - 2.)).abs() < delta)
            else:
                self.assertEqual(uvt[1], Scalar.MASKED)

        # Try all at once
        uvt = obs.uvt(uv_list, remask=True)
        self.assertEqual(uvt[0], [a[0] for a in uvt_list])
        self.assertEqual(uvt[1], [a[1] for a in uvt_list])

        ########################################################################
        # Old Slit unit tests
        ########################################################################

        Slit = TimedImage

        fov = FlatFOV((0.001,0.001), (10,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Slit(axes=('u','vt'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        tstep = indices.to_scalar(1)
        indices_ = indices.copy()
        indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

        # uvt() with remask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.time_at_tstep(tstep))
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        # uvt() with remask == True
        (uv, time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], cadence.tstride * indices.to_scalar(1)[:6])
        self.assertEqual(uv[:6].to_scalar(0), indices[:6].to_scalar(0))
        self.assertEqual(uv[:6].to_scalar(1), 0.5)

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices_.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices_.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time_max, time_min + cadence.texp)

        # uvt_range() with remask == True
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                             remask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.to_scalar(0)[:6], indices_.to_scalar(0)[:6])
        self.assertEqual(uv_min.to_scalar(1)[:6], 0)
        self.assertEqual(uv_max.to_scalar(0)[:6], indices_.to_scalar(0)[:6] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:6], 1)
        self.assertEqual(time_min[:6], cadence.tstride * indices_.to_scalar(1)[:6])
        self.assertEqual(time_max[:6], time_min[:6] + cadence.texp)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        tstep = indices.to_scalar(1)

        (time0, time1) = obs.time_range_at_uv(uv)
        uv_ = uv.copy()
        uv_.vals[:,0][uv_.vals[:,0] == 10] -= 1
        uv_.vals[:,1][uv_.vals[:,1] == 20] -= 1

        self.assertEqual(time0, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time1, time0 + cadence.texp)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 6*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:6], cadence.time_range_at_tstep(tstep)[0][:6])
        self.assertEqual(time1[:6], time0[:6] + cadence.texp)

        ####################################

        # Alternative axis order ('ut','v')

        fov = FlatFOV((0.001,0.001), (1,20))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = Slit(axes=('ut','v'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices_.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices_.vals[:,1] == 20] -= 1

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(0))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices_.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices_.to_scalar(1) + 1)
        self.assertEqual(time_min, cadence.tstride * indices_.to_scalar(0))
        self.assertEqual(time_max, time_min + cadence.texp)

        ####################################

        # Alternative texp for discontinuous indices

        fov = FlatFOV((0.001,0.001), (1,20))
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Slit(axes=('ut','v'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((0,0))[1],  0.)
        self.assertEqual(obs.uvt((5,0))[1], 50.)
        self.assertEqual(obs.uvt((5,5))[1], 50.)

        eps = 1.e-14
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

        # Test using scalar indices
        below = obs.uvt((0,20 - eps), remask=True)[0].to_scalar(1)
        exact = obs.uvt((0,20      ), remask=True)[0].to_scalar(1)
        above = obs.uvt((0,20 + eps), remask=True)[0].to_scalar(1)

        self.assertTrue(below < 20.)
        self.assertTrue(20. - below < delta)
        self.assertTrue(exact == 20.)
        self.assertTrue(above == Scalar.MASKED)
        self.assertTrue(above.mask)

        # Test using a Vector index
        indices = Vector([(0,20 - eps), (0,20), (0,20 + eps)])

        u = obs.uvt(indices, remask=True)[0].to_scalar(1)
        self.assertTrue(u == (below, exact, above))

        # Alternative texp and axes
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Slit(axes=('a','v','b','ut','c'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

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

        ########################################################################
        # Old RasterSlit unit tests
        ########################################################################

        RasterSlit = TimedImage

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=1., steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                         path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

        # uvt() with remask == False
        (uv, time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, [0, 100, 190, 10, 110, 200, 200])
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        # uvt() with remask == True
        (uv, time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], [0, 100, 190, 10, 110, 200])
        self.assertEqual(uv[:6].to_scalar(0), indices[:6].to_scalar(0))
        self.assertEqual(uv[:6].to_scalar(1), 0.5)

        # uvt() with remask == True, new indices
        non_ints = indices + (0.2, 0.9)
        (uv, time) = obs.uvt(non_ints, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(time.mask == uv.mask))

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, [0, 100, 190,  9, 109, 199, 199])
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        # uvt_range() with remask == True
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(non_ints,
                                                             remask=True)

        self.assertEqual(Boolean(uv_min.mask), 2*[False] + 5*[True])
        self.assertEqual(Boolean(uv_max.mask), uv_min.mask)
        self.assertEqual(Boolean(time_min.mask), uv_min.mask)
        self.assertEqual(Boolean(time_max.mask), uv_min.mask)

        self.assertEqual(uv_min.to_scalar(0)[:2], indices.to_scalar(0)[:2])
        self.assertEqual(uv_min.to_scalar(1)[:2], 0)
        self.assertEqual(uv_max.to_scalar(0)[:2], indices.to_scalar(0)[:2] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:2], 1)
        self.assertEqual(time_min[:2],
                         (slow_cadence.tstride * indices.to_scalar(1) +
                          fast_cadence.tstride * indices.to_scalar(0))[:2])
        self.assertEqual(time_max[:2], time_min[:2] + fast_cadence.texp)

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

        (time0, time1) = obs.time_range_at_uv(uv)

        self.assertEqual(time0, [0, 190, 9, 199, 199])
        self.assertEqual(time1, time0 + fast_cadence.texp)

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], [0, 190, 9, 199])
        self.assertEqual(time1[:4], time0[:4] + fast_cadence.texp)

        ####################################
        # Alternative axis order ('uslow','vfast')
        ####################################

        fov = FlatFOV((0.001,0.001), (1,20))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        fast_cadence = Metronome(tstart=0., tstride=0.5, texp=0.5, steps=20)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('uslow','vfast'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Pair([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])
        indices_ = indices.copy()
        indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
        indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

        (uv, time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), 0.5)
        self.assertEqual(uv.to_scalar(1), indices.to_scalar(1))
        self.assertEqual(time, [0, 5, 10, 90, 95, 100, 100])

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), 0)
        self.assertEqual(uv_min.to_scalar(1), indices_.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), 1)
        self.assertEqual(uv_max.to_scalar(1), indices_.to_scalar(1) + 1)
        self.assertEqual(time_min, [0, 5, 9.5, 90, 95, 99.5, 99.5])
        self.assertEqual(time_max, time_min + fast_cadence.texp)

        (time0, time1) = obs.time_range_at_uv(indices)

        self.assertEqual(time0, time_min)
        self.assertEqual(time1, time0 + fast_cadence.texp)

        ################################################
        # Alternative texp for discontinuous indices
        ################################################

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=8., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.5, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                                                 path='SSB', frame='J2000')

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
        self.assertTrue(abs(obs.uvt((6.2    ,1))[0] - (6.2,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.4    ,2))[0] - (6.4,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.6    ,3))[0] - (6.6,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((6.8    ,4))[0] - (6.8,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,6))[0] - (7.0,0.5)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ))[0] - (1.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((2, 1.25   ))[0] - (2.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((3, 2.5    ))[0] - (3.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((4, 3.75   ))[0] - (4.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5 - eps))[0] - (5.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((5, 5.     ))[0] - (5.,0.5)) < delta)

        ################################################
        # Alternative tstride for even more discontinuous indices
        ################################################

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=11., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('ufast','vslow'), cadence=cadence, fov=fov,
                                                 path='SSB', frame='J2000')

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
        pair = (10-eps, 0)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 9.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10, 0)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 9.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10+eps, 0)
        self.assertTrue(obs.uvt(pair, True)[0].mask)

        pair = (10-eps, 1-eps)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 9.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10, 1)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 20.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10, 20)
        self.assertTrue((obs.uvt(pair, True)[0] - (10, 0.5)).rms() < delta)
        self.assertTrue((obs.uvt(pair, True)[1] - 218.8).abs() < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10, 20+eps)
        self.assertTrue(obs.uvt(pair, True)[0].mask)

        ################################################
        # Alternative, discontinuous and weird axes
        ################################################

        fov = FlatFOV((0.001,0.001), (10,1))
        slow_cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        fast_cadence = Metronome(tstart=0., tstride=1., texp=0.8, steps=10)
        cadence = DualCadence(slow_cadence, fast_cadence)
        obs = RasterSlit(axes=('a','vslow','b','ufast','c'), cadence=cadence, fov=fov, path='SSB', frame='J2000')

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
        self.assertTrue(abs(obs.uvt((1,1,4,6.2    ,7))[0] - (6.2,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,2,4,6.4    ,7))[0] - (6.4,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,3,4,6.6    ,7))[0] - (6.6,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,4,4,6.8    ,7))[0] - (6.8,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1,6,4,7.     ,7))[0] - (7.0,0.5)) < delta)

        self.assertTrue(abs(obs.uvt((1, 0      ,4,1,7))[0] - (1.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 1.25   ,4,2,7))[0] - (2.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 2.5    ,4,3,7))[0] - (3.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 3.75   ,4,4,7))[0] - (4.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5 - eps,4,5,7))[0] - (5.,0.5)) < delta)
        self.assertTrue(abs(obs.uvt((1, 5.     ,4,5,7))[0] - (5.,0.5)) < delta)

        ########################################################################
        # Old Pushframe unit tests
        ########################################################################

        Pushframe = TimedImage

        flatfov = FlatFOV((0.001,0.001), (10,20))
        cadence = TDICadence(lines=20, tstart=100., tdi_texp=10., tdi_stages=2,
                             tdi_sign=-1)
        obs = Pushframe(axes=('u','vt'),
                        cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

        indices = Vector([( 0,0),( 0,1),( 0,10),( 0,18),( 0,19),( 0,20),( 0,21),
                          (10,0),(10,1),(10,10),(10,18),(10,19),(10,20),(10,21)])
        tstep = indices.to_scalar(1)

        # uvt() with remask == False
        (uv,time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(uv, Pair.as_pair(indices))
        self.assertEqual(time, 2*[100,100,100,100,110,120,120])

        # uvt() with remask == True
        (uv,time) = obs.uvt(indices, remask=True)

        self.assertTrue(np.all(uv.mask == np.array(2*(6*[False]+[True]))))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time, cadence.time_at_tstep(tstep, remask=True))
        self.assertEqual(uv[:6], Pair.as_pair(indices)[:6])

        # uvt_range() with remask == False
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min,
                         [(0,0),(0,1),(0,10),(0,18),(0,19),(0,19),(0,21),
                          (9,0),(9,1),(9,10),(9,18),(9,19),(9,19),(9,21)])
        self.assertEqual(uv_max, uv_min + (1,1))
        self.assertEqual(time_min, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time_max, cadence.time_range_at_tstep(tstep)[1])

        # uvt_range() with remask == False, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time_max, cadence.time_range_at_tstep(tstep)[1])

        # uvt_range() with remask == True, new indices
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                             remask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(5*[False] + 9*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], Pair.as_pair(indices)[:2])
        self.assertEqual(uv_max[:2], Pair.as_pair(indices)[:2] + (1,1))
        self.assertEqual(time_min[:2], cadence.time_range_at_tstep(tstep)[0][:2])
        self.assertEqual(time_max[:2], cadence.time_range_at_tstep(tstep)[1][:2])

        # time_range_at_uv() with remask == False
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])
        tstep = uv.to_scalar(1)

        (time0, time1) = obs.time_range_at_uv(uv)
        self.assertEqual(time0, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time1, cadence.time_range_at_tstep(tstep)[1])

        # time_range_at_uv() with remask == True
        (time0, time1) = obs.time_range_at_uv(uv, remask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], cadence.time_range_at_tstep(tstep)[0][:4])
        self.assertEqual(time1[:4], cadence.time_range_at_tstep(tstep)[1][:4])

        # Alternative axis order ('ut','v')
        cadence = TDICadence(lines=10, tstart=100., tdi_texp=10., tdi_stages=10,
                             tdi_sign=-1)
        obs = Pushframe(axes=('ut','v'),
                        cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

        indices = Vector([(-1,0),(0,-1),(0,0),(0,20),(9,0),(10,0),(11,0),(11,20)])
        tstep = indices.to_scalar(0)

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, Pair.as_pair(indices))
        self.assertEqual(time, cadence.time_at_tstep(tstep))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min,
                         [(-1,0),(0,-1),(0,0),(0,19),(9,0),(9,0),(11,0),(11,19)])
        self.assertEqual(uv_max, uv_min + (1,1))
        self.assertEqual(time_min, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time_max, cadence.time_range_at_tstep(tstep)[1])

        (time0,time1) = obs.time_range_at_uv(indices)

        self.assertEqual(time0, cadence.time_range_at_tstep(tstep)[0])
        self.assertEqual(time1, cadence.time_range_at_tstep(tstep)[1])

        # Alternative texp for discontinuous indices
        cadence = TDICadence(lines=10, tstart=100., tdi_texp=10., tdi_stages=10,
                                       tdi_sign=1)
        obs = Pushframe(axes=('ut','v'),
                        cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[0], 100.)

        self.assertEqual(obs.uvt((-1,0))[0], (-1,0))
        self.assertEqual(obs.uvt(( 0,0))[0], ( 0,0))
        self.assertEqual(obs.uvt(( 5,0))[0], ( 5,0))
        self.assertEqual(obs.uvt(( 5,5))[0], ( 5,5))
        self.assertEqual(obs.uvt(( 9,5))[0], ( 9,5))
        self.assertEqual(obs.uvt((9.5,5))[0],(9.5,5))
        self.assertEqual(obs.uvt((10,5))[0], (10,5))

        self.assertEqual(obs.uvt((-1,0))[1], 190.)
        self.assertEqual(obs.uvt(( 0,0))[1], 190.)
        self.assertEqual(obs.uvt(( 5,0))[1], 140.)
        self.assertEqual(obs.uvt(( 5,5))[1], 140.)
        self.assertEqual(obs.uvt(( 9,5))[1], 100.)
        self.assertEqual(obs.uvt((9.5,5))[1],150.)
        self.assertEqual(obs.uvt((10,5))[1], 200.)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
