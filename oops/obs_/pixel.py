################################################################################
# oops/obs_/pixel.py: Subclass Pixel of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.obs_.observation import Observation
from oops.path_.path       import Path
from oops.frame_.frame     import Frame
from oops.cadence_.cadence import Cadence
from oops.cadence_.instant import Instant
from oops.event            import Event

#*******************************************************************************
# Pixel
#*******************************************************************************
class Pixel(Observation):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A Pixel is a subclass of Observation consisting of one or more
    measurements obtained from a single rectangular pixel.

    Generalization to other pixel shapes is TDB. 7/24/12 MRS
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['axes', 'cadence', 'fov', 'path', 'frame', '**subfields']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Pixel observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 't' should
                        appear at the location of the array's time-axis.

            cadence     a Cadence object defining the start time and duration of
                        each consecutive measurement. Alternatively, a dictionary 
                        containing the following entries, from which a cadence 
                        object is constructed:

                         tbd:    TBD


            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a Pixel object, both axes of the
                        FOV must have length 1.

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
        self.fov = fov
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        #--------------------------------------------------
        # Axes
        #--------------------------------------------------
        self.axes = list(axes)
        self.u_axis = -1
        self.v_axis = -1
        self.swap_uv = False
        if 't' in self.axes:
            self.t_axis = self.axes.index('t')
        else:
            self.t_axis = -1

        #--------------------------------------------------
        # Cadence
        #--------------------------------------------------
	if isinstance(cadence, Cadence): self.cadence = cadence
	else: self.cadence = self._default_cadence(cadence)

        #--------------------------------------------------
        # Timing
        #--------------------------------------------------
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime
        self.scalar_times = (Scalar(self.time[0]), Scalar(self.time[1]))

        #--------------------------------------------------
        # Shape / Size
        #--------------------------------------------------
        assert self.fov.uv_shape == (1,1)
        self.uv_shape = (1,1)

        shape_list = len(axes) * [0]
        if self.t_axis >= 0:
            shape_list[self.t_axis] = self.cadence.shape[0]
        self.shape = tuple(shape_list)

        #--------------------------------------------------
        # Optional subfields
        #--------------------------------------------------
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        return
    #===========================================================================



    #===========================================================================
    # _default_cadence
    #===========================================================================
    def _default_cadence(self, dict):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a cadence object a dictionary of parameters.

        Input:
            dict        Dictionary containing the following entries:

                         tstart: Observation start time.
                         nexp:   Number of exposures in the observation.
                         exp:    Exposure time for each observation.

        Return:         Cadence object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        tbd = dict['tbd']
        return Instant(tbd)
    #===========================================================================



    ############################################################################

    #===========================================================================
    # uvt
    #===========================================================================
    def uvt(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values.

        Input:
            indices     a 1-D Vector of array indices.
            fovmask     True to mask values outside the field of view.

        Return:         (uv, time)
            uv          a Pair defining the values of (u,v) associated with the
                        array indices.
            time        a Scalar defining the time in seconds TDB associated
                        with the array indices.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        indices = Vector.as_vector(indices)
        tstep = indices.to_scalar(self.t_axis)

        time = self.cadence.time_at_tstep(tstep, mask=fovmask)
        uv = Pair((0.5,0.5))

        if fovmask:
            is_inside = self.cadence.time_is_inside(time, inclusive=True)
            if not is_inside.all():
                mask = (indices.mask | is_inside.logical_not().values
                                     | is_inside.mask)
                time = Scalar(time, mask)

                uv_vals = np.empty(indices.shape + (2,))
                uv_vals[...] = 0.5
                uv = Pair(uv_vals, mask)

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
        indices = Vector.as_vector(indices).as_int()
        tstep = indices.to_scalar(self.t_axis)

        (time_min,
         time_max) = self.cadence.time_range_at_tstep(tstep, mask=fovmask)
        uv_min = Pair.ZEROS
        uv_max = Pair.ONES

        if fovmask:
            mask = time_min.mask
            if np.any(mask):
                uv_min_vals = np.zeros(indices.shape + (2,))
                uv_max_vals = np.ones(indices.shape  + (2,))

                uv_min = Pair(uv_min_vals, mask)
                uv_max = Pair(uv_max_vals, mask)

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
        return (Pair.ZEROS, Pair.ONES)
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
        if not fovmask: return self.scalar_times

        uv_pair = Pair.as_pair(uv_pair)
        is_outside = self.uv_is_outside(uv_pair, inclusive=True)
        if np.any(is_outside):
            mask = uv_pair.mask | is_outside

            time0_vals = np.empty(uv_pair.shape)
            time1_vals = np.empty(uv_pair.shape)

            time0_vals[...] = self.time[0]
            time1_vals[...] = self.time[1]

            time0 = Scalar(time0_vals, mask)
            time1 = Scalar(time1_vals, mask)

        return (time0, time1)
    #===========================================================================



    #===========================================================================
    # sweep_duv_dt
    #===========================================================================
    def sweep_duv_dt(self, uv_pair):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the mean local sweep speed of the instrument along (u,v) axes.


        Input:
            uv_pair     a Pair of spatial indices (u,v).

        Return:         a Pair containing the local sweep speed in units of
                        pixels per second in the (u,v) directions.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Pair.ZEROS
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
        obs = Pixel(self.axes, self.cadence.time_shift(dtime),
                    self.fov, self.path, self.frame)

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs
    #===========================================================================



    ############################################################################
    # Overrides of Observation class methods
    ############################################################################

    #===========================================================================
    # event_at_grid
    #===========================================================================
    def event_at_grid(self, meshgrid, time=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Returns an event object describing the arrival of a photon at a set
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert len(self.cadence.shape) == 1

        if time is None:
            tstep = np.arange(self.cadence.shape[0]) + 0.5
            time = self.cadence.time_at_tstep(tstep)
            time = time.append_axes(len(meshgrid.shape))

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        #---------------------------------------------------------------------
        # Insert the arrival directions
        #---------------------------------------------------------------------
        event.neg_arr_ap = meshgrid.los

        return event
    #===========================================================================



    #===========================================================================
    # gridless_event
    #===========================================================================
    def gridless_event(self, meshgrid, time=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Returns an event object describing the arrival of a photon at a set
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert len(self.cadence.shape) == 1

        if time is None:
            tstep = np.arange(self.cadence.shape[0]) + 0.5
            time = self.cadence.time_at_tstep(tstep)
            time = time.append_axes(len(meshgrid.shape))

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        return event
    #===========================================================================


#*******************************************************************************



################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_Pixel
#*******************************************************************************
class Test_Pixel(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        from oops.cadence_.metronome import Metronome
        from oops.fov_.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (1,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pixel(axes=('t'),
                    cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,),(1,),(20,),(21,)])

        #----------------------------------
        # uvt() with fovmask == False
        #----------------------------------
        (uv,time) = obs.uvt(indices)

        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, cadence.tstride * indices.to_scalar(0))
        self.assertEqual(uv, (0.5,0.5))

        #-----------------------------------
        # uvt() with fovmask == True
        #-----------------------------------
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:3], cadence.tstride * indices.to_scalar(0)[:3])
        self.assertEqual(uv[:3], (0.5,0.5))

        #--------------------------------------
        # uvt_range() with fovmask == False
        #--------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))
        self.assertEqual(time_min, indices.to_scalars()[0] * cadence.tstride)
        self.assertEqual(time_max, time_min + cadence.texp)

        #-----------------------------------------------------
        # uvt_range() with fovmask == False, new indices
        #-----------------------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices + (0.2,))

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))
        self.assertEqual(time_min, indices.to_scalars()[0] * cadence.tstride)
        self.assertEqual(time_max, time_min + cadence.texp)

        #----------------------------------------------------
        # uvt_range() with fovmask == True, new indices
        #----------------------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices + (0.2,),
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:2], (0,0))
        self.assertEqual(uv_max[:2], (1,1))
        self.assertEqual(time_min[:2], indices.to_scalars()[0][:2] *
                                       cadence.tstride)
        self.assertEqual(time_max[:2], time_min[:2] + cadence.texp)

        #-----------------------------------------
        # times_at_uv() with fovmask == False
        #-----------------------------------------
        uv = Pair([(0,0),(0,1),(1,0),(1,1),(1,2)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, obs.time[0])
        self.assertEqual(time1, obs.time[1])

        #-----------------------------------------
        # times_at_uv() with fovmask == True
        #-----------------------------------------
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:4], obs.time[0])
        self.assertEqual(time1[:4], obs.time[1])

        ####################################


        #--------------------------------------
        # Alternative axis order ('a','t')
        #--------------------------------------

        fov = FlatFOV((0.001,0.001), (1,1))
        cadence = Metronome(tstart=0., tstride=10., texp=10., steps=20)
        obs = Pixel(axes=('a','t'),
                    cadence=cadence, fov=fov, path='SSB', frame='J2000')

        indices = Vector([(0,0),(1,1),(0,20,),(1,21)])

        #-----------------------------------
        # uvt() with fovmask == False
        #-----------------------------------
        (uv,time) = obs.uvt(indices)

        self.assertFalse(uv.mask)
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time.without_mask(),
                         cadence.tstride * indices.to_scalar(1))
        self.assertEqual(uv, (0.5,0.5))

        #----------------------------------
        # uvt() with fovmask == True
        #----------------------------------
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:3], cadence.tstride * indices[:3].to_scalar(1))
        self.assertEqual(uv[:3], (0.5,0.5))

        #----------------------------------------
        # uvt_range() with fovmask == False
        #----------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        #----------------------------------------------------
        # uvt_range() with fovmask == False, new indices
        #----------------------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

        self.assertFalse(uv_min.mask)
        self.assertFalse(uv_max.mask)
        self.assertFalse(time_min.mask)
        self.assertFalse(time_max.mask)

        self.assertEqual(uv_min, (0,0))
        self.assertEqual(uv_max, (1,1))
        self.assertEqual(time_min, cadence.tstride * indices.to_scalar(1))
        self.assertEqual(time_max, time_min + cadence.texp)

        #--------------------------------------------------
        # uvt_range() with fovmask == True, new indices
        #--------------------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices + (0.2,0.2),
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min[:3], (0,0))
        self.assertEqual(uv_max[:3], (1,1))
        self.assertEqual(time_min[:3], cadence.tstride *
                                       indices.to_scalar(1)[:3])
        self.assertEqual(time_max[:3], time_min[:3] + cadence.texp)

        #-----------------------------------------
        # times_at_uv() with fovmask == False
        #-----------------------------------------
        uv = Pair([(0,0),(0,1),(1,0),(1,1),(1,2)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, obs.time[0])
        self.assertEqual(time1, obs.time[1])

        #----------------------------------------
        # times_at_uv() with fovmask == True
        #----------------------------------------
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 4*[False] + [True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:4], obs.time[0])
        self.assertEqual(time1[:4], obs.time[1])
    #===========================================================================


#*******************************************************************************


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
