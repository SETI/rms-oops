################################################################################
# oops/observation/pushframe.py: Subclass Pushframe of class Observation
################################################################################

import numpy as np
from polymath import Pair, Vector, Qube

from oops.observation           import Observation
from oops.observation.snapshot  import Snapshot
from oops.cadence.tdicadence    import TDICadence
from oops.fov.tdifov            import TDIFOV
from oops.frame                 import Frame
from oops.path                  import Path

class Pushframe(Observation):
    """An Observation obtained with a TDI ("Time Delay and Integration") camera.

    It is a 2-D image made up of lines of pixels, each exposed and shifted
    progressively to track a scene moving through the FOV at a constant rate.
    """

    INVENTORY_IMPLEMENTED = True

    #===========================================================================
    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        """Constructor for a Pushframe.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'u' or 'ut'
                        should appear at the location of the array's u-axis;
                        'vt' or 'v' should appear at the location of the array's
                        v-axis. The 't' suffix is used for the one of these axes
                        that is swept by the time-delayed integration.

            cadence     a TDICadence object defining the start time and duration
                        of each consecutive line of the detector. Alternatively,
                        a tuple or dictionary providing input arguments to the
                        TDICadence constructor (after the number of lines, which
                        is defined by the FOV):
                            (tstart, tdi_texp, tdi_stages[, tdi_sign])

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
        assert (('u' in self.axes and 'vt' in self.axes) or
                ('v' in self.axes and 'ut' in self.axes))

        if 'ut' in self.axes:
            self.u_axis = self.axes.index('ut')
            self.v_axis = self.axes.index('v')
            self.t_axis = self.u_axis
            self._tdi_uv_index = 0
        else:
            self.u_axis = self.axes.index('u')
            self.v_axis = self.axes.index('vt')
            self.t_axis = self.v_axis
            self._tdi_uv_index = 1

        self.swap_uv = (self.u_axis > self.v_axis)

        # Shape / Size
        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        # Cadence
        lines = self.uv_shape[self._tdi_uv_index]

        if isinstance(cadence, (tuple,list)):
            self.cadence = TDICadence(lines, *cadence)
        elif isinstance(cadence, dict):
            self.cadence = TDICadence(lines, **cadence)
        elif isinstance(cadence, TDICadence):
            self.cadence = cadence
            assert self.cadence.shape == (lines,)
        else:
            raise TypeError('Invalid cadence class for PushFrame: ' +
                            type(cadence).__name__)

        # Timing
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        # Snapshot class proxy (for inventory)
        replacements = {
            'ut':  'u',
            'vt':  'v',
        }

        snapshot_axes = [replacements.get(axis, axis) for axis in axes]
        snapshot_tstart = self.cadence.time[0]
        snapshot_texp = self.cadence.time[1] - self.cadence.time[0]

        self.snapshot = Snapshot(snapshot_axes, snapshot_tstart, snapshot_texp,
                                 self.fov, self.path, self.frame, **subfields)

    def __getstate__(self):
        return (self.axes, self.cadence, self.fov, self.path, self.frame,
                self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])

    #===========================================================================
    def uvt(self, indices, remask=False):
        """Coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a Scalar or Vector of array indices.
            remask      True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) within the FOV that
                        are associated with the array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """

        indices = Vector.as_vector(indices)
        uv = indices.to_pair((self.u_axis, self.v_axis))

        tstep = uv.to_scalar(self._tdi_uv_index)
        time = self.cadence.time_at_tstep(tstep, remask=remask)

        # Apply mask if necessary
        if remask:
            is_outside = self.uv_is_outside(uv, inclusive=True)
            if is_outside.any_true_or_masked():
                uv = uv.mask_where(is_outside)
                time = time.mask_where(is_outside)

        return (uv, time)

    #===========================================================================
    def uvt_range(self, indices, remask=False):
        """Ranges of FOV coordinates and time for integer array indices.

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

        indices = Vector.as_vector(indices)

        uv = indices.to_pair((self.u_axis,self.v_axis))
        uv_min = uv.int(top=self.uv_shape, remask=remask)

        tstep = uv_min.to_scalar(self._tdi_uv_index)
        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep,
                                                                remask=remask)

        # Apply mask if necessary
        if remask:
            is_outside = self.uv_is_outside(uv_min, inclusive=True)
            if is_outside.any_true_or_masked():
                uv_min = uv_min.mask_where(is_outside)
                time_min = time_min.mask_where(is_outside)
                # Note that time_max is a single value so no mask is needed

        return (uv_min, uv_min + Pair.INT11, time_min, time_max)

    #===========================================================================
    def uv_range_at_tstep(self, tstep, remask=False):
        """A tuple defining the range of FOV (u,v) coordinates active at a
        particular time step.

        Input:
            tstep       a Scalar step index.
            remask      True to mask values outside the time interval.

        Return:         a tuple (uv_min, uv_max)
            uv_min      a Pair defining the minimum values of FOV (u,v)
                        coordinates active at this time step.
            uv_min      a Pair defining the maximum values of FOV (u,v)
                        coordinates active at this time step (exclusive).
        """

        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep,
                                                                remask=remask)
        ### Joe--Please fix. Pushframe.uv_range_at_tstep needs to account for the
        ### fact that the last lines have a later start time, meaning
        ### that the (u,v) range varies.

        return (Pair.INT00, Pair.as_pair(self.shape))

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

        tstep = Pair.as_pair(uv_pair).to_scalar(self._tdi_uv_index)
#         tstep_int = tstep.int(top=self.uv_shape, remask=remask)
#         (time_min, time_max) = self.cadence.time_range_at_tstep(tstep_int,
#                                                                 remask=False)
        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep,
                                                                remask=False)

        # Apply mask if necessary
        if remask:
            is_outside = self.uv_is_outside(uv_pair, inclusive=True)
            if is_outside.any_true_or_masked():
                time_min = time_min.mask_where(is_outside)
                # time1 is shapeless so it can't take a mask

        return (time_min, time_max)

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

        return Observation.uv_range_at_time_1d(self, time, remask,
                                               self._tdi_uv_index)

    #===========================================================================
    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """

        obs = Pushframe(self.axes, self.cadence.time_shift(dtime),
                        self.fov, self.path, self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs

    #===========================================================================
    def inventory(self, *args, **kwargs):
        """Info about the bodies that appear unobscured inside the FOV. See
        Snapshot.inventory() for details.

        WARNING: Not properly updated for class PushFrame. Use at your own risk.
        This operates by returning every body that would have been inside the
        FOV of this observation if it were instead a Snapshot, evaluated at the
        given tfrac.
        """

        return self.snapshot.inventory(*args, **kwargs)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Pushframe(unittest.TestCase):

    def runTest(self):

        from oops.cadence.tdicadence import TDICadence
        from oops.fov.flatfov import FlatFOV

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
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
