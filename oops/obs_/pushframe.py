################################################################################
# oops/obs_/pushframe.py: Subclass Pushframe of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation   import Observation
from oops.obs_.snapshot      import Snapshot
from oops.cadence_.cadence   import Cadence
from oops.cadence_.metronome import Metronome
from oops.path_.path         import Path
from oops.frame_.frame       import Frame
from oops.body               import Body
from oops.event              import Event

#*******************************************************************************
# Pushframe
#*******************************************************************************
class Pushframe(Observation):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A Pushframe is an Observation obtained with a TDI ("Time Delay and
    Integration") camera. It is a 2-D image made up of lines of pixels, each
    exposed and shifted progressively to track a scene moving through the FOV at
    a constant rate.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    INVENTORY_IMPLEMENTED = True

    PACKRAT_ARGS = ['axes', 'cadence', 'fov', 'path', 'frame', '**subfields']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Pushframe.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'u' or 'ut'
                        should appear at the location of the array's u-axis;
                        'vt' or 'v' should appear at the location of the array's
                        v-axis. The 't' suffix is used for the one of these axes
                        that is swept by the time-delayed integration.

            cadence     a TdiCadence object defining the start time and duration
                        of each consecutive line of the detector. Alternatively,
                        a tuple or dictionary providing input arguments to the
                        TdiCadence constructor (after the number of lines, which
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #--------------------------------------------------
        # Basic properties
        #--------------------------------------------------
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        #--------------------------------------------------
        # FOV
        #--------------------------------------------------
        self.fov = fov
        self.uv_shape = tuple(self.fov.uv_shape.vals)

        #--------------------------------------------------
        # Axes
        #--------------------------------------------------
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

        #--------------------------------------------------
        # Shape / Size
        #--------------------------------------------------
        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        #--------------------------------------------------
        # Cadence
        #--------------------------------------------------
        lines = self.uv_shape[self._tdi_uv_index]

        if isinstance(cadence, (tuple,list)):
            self.cadence = TdiCadence(lines, *cadence)
        elif isinstance(cadence, dict):
            self.cadence = TdiCadence(lines, **cadence)
        elif isinstance(cadence, TdiCadence):
            self.cadence = cadence
            assert self.cadence.shape == (lines,)
        else:
            raise TypeError('Invalid cadence class for PushFrame: ' +
                            type(cadence).__name__)

        #--------------------------------------------------
        # Timing
        #--------------------------------------------------
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

#        #---------------------------------------------------------------
#        # Fractional timing
#        #  For each pixel, determine the fractional window within each
#        #  exposure that corresponds to the scene geometry
#        #---------------------------------------------------------------
#         time0 = self.cadence.time_at_tstep(0)
#         time1 = time0 + self.cadence.texp[self.cadence.steps-1]
#
#         times = self.cadence.time_range_at_tstep(
#                                        np.indices([self.shape[self.t_axis]]))
#         dtimes = times[1] - times[0]
#
#         tfrac0 = ((time0 - times[0])/dtimes).vals.T
#         tfrac1 = ((time1 - times[0])/dtimes).vals.T
#         self.tfrac = ( Scalar.as_scalar(np.broadcast_to(tfrac0, self.shape)),
#                        Scalar.as_scalar(np.broadcast_to(tfrac1, self.shape)) )

        #--------------------------------------------------
        # Optional subfields
        #--------------------------------------------------
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        #--------------------------------------------------
        # Snapshot class proxy (for inventory)
        #--------------------------------------------------
        replacements = {
            'ut':  'u',
            'vt':  'v',
        }

        snapshot_axes = [replacements.get(axis, axis) for axis in axes]
        snapshot_tstart = self.cadence.time[0]
        snapshot_texp = self.cadence.time[1] - self.cadence.time[0]

        self.snapshot = Snapshot(snapshot_axes, snapshot_tstart, snapshot_texp,
                                 self.fov, self.path, self.frame, **subfields)
    #===========================================================================



    #===========================================================================
    # uvt
    #===========================================================================
    def uvt(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return coordinates (u,v) and time t for indices into the data array.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        indices = Vector.as_vector(indices)
        uv = indices.to_pair((self.u_axis,self.v_axis))

        tstep = uv.to_scalar(self._tdi_uv_index)
        time = self.cadence.time_at_tstep(tstep, mask=fovmask)

        #------------------------------
        # Apply mask if necessary
        #------------------------------
        if fovmask:
            is_outside = self.uv_is_outside(uv, inclusive=True)
            new_mask = time.mask | is_outside
            if np.any(is_outside):
                uv = uv.mask_where(is_outside)
                time = time.mask_where(is_outside)

        return (uv, time)
    #===========================================================================



    #===========================================================================
    # uvt_range
    #===========================================================================
    def uvt_range(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return ranges of coordinates and time for integer array indices.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        indices = Vector.as_vector(indices)

        uv = indices.to_pair((self.u_axis,self.v_axis))
        uv_min = uv.as_int()
        uv_max = uv_min + Pair.ONES

        tstep = uv_min.to_scalar(self._tdi_uv_index)
        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep,
                                                                mask=fovmask)

        #------------------------------
        # Apply mask if necessary
        #------------------------------
        if fovmask:
            is_outside = self.uv_is_outside(uv_min, inclusive=False)
            new_mask = time_min.mask | is_outside
            if np.any(new_mask):
                uv_min = uv_min.mask_where(new_mask)
                uv_max = uv_max.mask_where(new_mask)
                time_min = time_min.mask_where(new_mask)
                # Note that time_max is a single value so no mask is needed

        return (uv_min, uv_max, time_min, time_max)
    #===========================================================================



    #===========================================================================
    # uv_range_at_tstep
    #===========================================================================
    def uv_range_at_tstep(self, *tstep):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a tuple defining the range of (u,v) coordinates active at a
        particular time step.

        Input:
            tstep       a time step index (one or two integers).

        Return:         a tuple (uv_min, uv_max)
            uv_min      a Pair defining the minimum values of (u,v) coordinates
                        active at this time step.
            uv_min      a Pair defining the maximum values of (u,v) coordinates
                        active at this time step (exclusive).
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return (Pair.ZERO, Pair.as_pair(self.shape))
    #===========================================================================



    #===========================================================================
    # times_at_uv
    #===========================================================================
    def times_at_uv(self, uv_pair, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return start and stop times of the specified spatial pixel (u,v).

        Input:
            uv_pair     a Pair of spatial (u,v) coordinates in and observation's
                        field of view. The coordinates need not be integers, but
                        any fractional part is truncated.
            fovmask     True to mask values outside the field of view.

        Return:         a tuple containing Scalars of the start time and stop
                        time of each (u,v) pair, as seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        tstep = Pair.as_pair(uv_pair).to_scalar(self._tdi_uv_index).as_int()
        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep,
                                                                mask=fovmask)

        #------------------------------
        # Apply mask if necessary
        #------------------------------
        if fovmask:
            is_outside = self.uv_is_outside(uv_pair, inclusive=True)
            if np.any(is_outside):
                time_min = time_min.mask_where(is_outside)
                # time1 is shapeless so it can't take a mask

        return (time_min, time_max)
    #===========================================================================



    #===========================================================================
    # time_shift
    #===========================================================================
    def time_shift(self, dtime):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a copy of the observation object with a time-shift.

        Input:
            dtime       the time offset to apply to the observation, in units of
                        seconds. A positive value shifts the observation later.

        Return:         a (shallow) copy of the object with a new time.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obs = Pushframe(self.axes, self.cadence.time_shift(dtime),
                        self.fov, self.path, self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs
    #===========================================================================



    #===========================================================================
    # inventory
    #===========================================================================
    def inventory(*args, **kwargs):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the body names that appear unobscured inside the FOV. See
        Snapshot.inventory() for details.

        WARNING: Not properly updated for class PushFrame. Use at your own risk.
        This operates by returning every body that would have been inside the
        FOV of this observation if it were instead a Snapshot, evaluated at the
        given tfrac.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.snapshot.inventory(*args, **kwargs)
    #===========================================================================

#*******************************************************************************



################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_Pushframe
#*******************************************************************************
class Test_Pushframe(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):
        return          #!!!!!!!!! remove this!!!

        from oops.cadence_.metronome import Metronome
        from oops.fov_.flatfov import FlatFOV

        flatfov = FlatFOV((0.001,0.001), (10,20))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pushframe(axes=('u','vt'),
                        cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        #-------------------------------
        # uvt() with fovmask == False
        #-------------------------------
        (uv,time) = obs.uvt(indices)







        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertTrue(time.max() <= cadence.midtime)
        self.assertEqual(uv, Pair.as_pair(indices))

        #-------------------------------
        # uvt() with fovmask == True
        #-------------------------------
        (uv,time) = obs.uvt(indices, fovmask=True)

        return      #####################################


        self.assertTrue(np.all(uv.mask == np.array(6*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:6], cadence.tstride * indices.to_scalar(1)[:6])
        self.assertEqual(uv[:6], Pair.as_pair(indices)[:6])

        #--------------------------------------
        # uvt_range() with fovmask == False
        #--------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        #----------------------------------------------------
        # uvt_range() with fovmask == False, new indices
        #----------------------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        #--------------------------------------------------
        # uvt_range() with fovmask == True, new indices
        #--------------------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 5*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], Pair.as_pair(indices)[:2])
        self.assertEqual(uv_max[:2], Pair.as_pair(indices)[:2] + (1,1))
        self.assertEqual(time_min[:2], cadence.tstride *
                                       indices.to_scalar(1)[:2])
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        #------------------------------------------
        # times_at_uv() with fovmask == False
        #------------------------------------------
        uv = Pair([(0,0),(0,20),(10,0),(10,20),(10,21)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, cadence.tstride * uv.to_scalar(1))
        self.assertEqual(time1, time0 + cadence.texp)

        #-----------------------------------------
        # times_at_uv() with fovmask == True
        #-----------------------------------------
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == 4*[False] + [True]))
        self.assertEqual(time0[:4], cadence.tstride * uv.to_scalar(1)[:4])
        self.assertEqual(time1[:4], time0[:4] + cadence.texp)

        #----------------------------------------
        # Alternative axis order ('ut','v')
        #----------------------------------------
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=10)
        obs = Pushframe(axes=('ut','v'),
                        cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(0,10),(0,20),(10,0),(10,10),(10,20),(10,21)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv, Pair.as_pair(indices))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(0))

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min, Pair.as_pair(indices))
        self.assertEqual(uv_max, Pair.as_pair(indices) + (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(0))
        self.assertEqual(time_max, time_min + cadence.texp)

        (time0,time1) = obs.times_at_uv(indices)

        self.assertEqual(time0, cadence.tstride * uv.to_scalar(0))
        self.assertEqual(time1, time0 + cadence.texp)

        #-----------------------------------------------------------
        # Alternative uv_size and texp for discontinuous indices
        #-----------------------------------------------------------
        cadence = Metronome(tstart=0., tstride=10., texp=8., steps=10)
        obs = Pushframe(axes=('ut','v'),
                        cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

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

        #--------------------------
        # Test the upper edge
        #--------------------------
        pair = (10-eps,20-eps)
        self.assertTrue(abs(obs.uvt(pair, True)[0].vals[0] -  9.5) < delta)
        self.assertTrue(abs(obs.uvt(pair, True)[0].vals[1] - 19.8) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10,20-eps)
        self.assertTrue(abs(obs.uvt(pair, True)[0].vals[0] -  9.5) < delta)
        self.assertTrue(abs(obs.uvt(pair, True)[0].vals[1] - 19.8) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10-eps,20)
        self.assertTrue(abs(obs.uvt(pair, True)[0].vals[0] -  9.5) < delta)
        self.assertTrue(abs(obs.uvt(pair, True)[0].vals[1] - 19.8) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        pair = (10,20)
        self.assertTrue(abs(obs.uvt(pair, True)[0].vals[0] -  9.5) < delta)
        self.assertTrue(abs(obs.uvt(pair, True)[0].vals[1] - 19.8) < delta)
        self.assertFalse(obs.uvt(pair, True)[0].mask)

        self.assertTrue(obs.uvt((10+eps,20), True)[0].mask)
        self.assertTrue(obs.uvt((10,20+eps), True)[0].mask)

        #----------------------
        # Try all at once
        #----------------------
        indices = Pair([(10-eps,20-eps), (10,20-eps), (10-eps,20), (10,20),
                        (10+eps,20), (10,20+eps)])

        (uv,t) = obs.uvt(indices, fovmask=True)
        self.assertTrue(np.all(t.mask == np.array(4*[False] + 2*[True])))

        #-------------------------------------------------
        # Alternative with uv_size and texp and axes
        #-------------------------------------------------
        obs = Pushframe(axes=('a','v','b','ut','c'),
                        cadence=cadence, fov=flatfov, path='SSB', frame='J2000')

        self.assertEqual(obs.time[1], 98.)

        self.assertEqual(obs.uvt((1,0,3,0,4))[1],  0.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1], 50.)
        self.assertEqual(obs.uvt((1,0,3,5,4))[1], 50.)

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((1,0,0,6      ,0))[1] - 60.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.25   ,0))[1] - 62.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.5    ,0))[1] - 64.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,6.75   ,0))[1] - 66.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7 - eps,0))[1] - 68.) < delta)
        self.assertTrue(abs(obs.uvt((1,0,0,7.     ,0))[1] - 70.) < delta)

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
    #===========================================================================


#*******************************************************************************


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
