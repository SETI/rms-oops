################################################################################
# oops/obs_/slit1d.py: Subclass Slit1D of class Observation
################################################################################

import numpy as np
from polymath import *

from oops.cadence_.metronome import Metronome
from oops.obs_.observation   import Observation
from oops.cadence_.metronome import Metronome
from oops.path_.path         import Path
from oops.frame_.frame       import Frame
from oops.event              import Event

#*******************************************************************************
# Slit1D
#*******************************************************************************
class Slit1D(Observation):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A Slit1D is subclass of Observation consisting of a 1-D slit measurement
    with no time-dependence. However, it may still have additional axes.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#    PACKRAT_ARGS = ['axes', 'det_size', 'cadence', 'fov', 'path',
#                    'frame', '**subfields']
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
    PACKRAT_ARGS = ['axes', 'det_size', 'tstart', 'texp', 'fov', 'path',
                    'frame', '**subfields']
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

    #===========================================================================
    # __init__
    #===========================================================================
#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#    def __init__(self, axes, det_size, cadence, fov, path, frame,
#                       **subfields):
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
    def __init__(self, axes, det_size, tstart, texp, fov, path, frame,
                       **subfields):
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Slit1D observation.

        Input:
            axes        a list or tuple of strings, with one value for each axis
                        in the associated data array. A value of 'u' should
                        appear at the location of the array's u-axis if any;
                        'v' should appear at the location of the array's v-axis
                        if any. Only one of 'u' or 'v' can appear in a Slit1D.

            det_size    the size of the detectors in FOV units parallel to the
                        slit. It will be < 1 if there are gaps between the
                        detectors.

#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#            cadence     a Cadence object defining the start time and duration 
#                        of the slit1d.  Alternatively, a tuple of the form:
#
#                          (tstart, texp)
#
#                        with:
#
#                          tstart: Observation start time.
#                          texp:   Exposure time for the observation.
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
            tstart      the start time of the observation in seconds TDB.

            texp        exposure time of the observation in seconds.
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-




            fov         a FOV (field-of-view) object, which describes the field
                        of view including any spatial distortion. It maps
                        between spatial coordinates (u,v) and instrument
                        coordinates (x,y). For a Slit1D object, one of the axes
                        of the FOV must have length 1.

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
        # Shape / Size
        #--------------------------------------------------
        self.shape = len(axes) * [0]

        self.det_size = det_size
        self.slit_is_discontinuous = (self.det_size < 1)

        #--------------------------------------------------
        # Axes
        #--------------------------------------------------
        self.axes = list(axes)
        assert (('u' in self.axes and 'v' not in self.axes) or
                ('v' in self.axes and 'u' not in self.axes))

        if 'u' in self.axes:
            self.u_axis = self.axes.index('u')
            self.v_axis = -1
            self.along_slit_index = self.u_axis
            self.along_slit_uv_axis = 0
            self.cross_slit_uv_axis = 1
            self.shape[self.u_axis] = self.fov.uv_shape.vals[0]
            self.along_slit_shape = self.shape[self.u_axis]
        else:
            self.u_axis = -1
            self.v_axis = self.axes.index('v')
            self.along_slit_index = self.v_axis
            self.along_slit_uv_axis = 1
            self.cross_slit_uv_axis = 0
            self.shape[self.v_axis] = self.fov.uv_shape.vals[1]
            self.along_slit_shape = self.shape[self.v_axis]

        self.swap_uv = False

        self.uv_shape = self.fov.uv_shape.vals
        assert self.fov.uv_shape.vals[self.cross_slit_uv_axis] == 1

        #--------------------------------------------------
        # Cadence
        #--------------------------------------------------
#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#        if isinstance(cadence, Cadence): self.cadence = cadence
#        else: self.cadence = self._default_cadence(*cadence)
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
        self.cadence = Metronome(tstart, texp, texp, 1)
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
	
        #--------------------------------------------------
        # Timing
        #--------------------------------------------------
        self.tstart = tstart
        self.texp = texp
        self.t_axis = -1
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self.scalar_time = (Scalar(self.time[0]), Scalar(self.time[1]))
        self.scalar_midtime = Scalar(self.midtime)

        #--------------------------------------------------
        # Optional subfields
        #--------------------------------------------------
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])
    #===========================================================================



    #===========================================================================
    # _default_cadence
    #===========================================================================
    def _default_cadence(self, tstart, texp):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a cadence object a dictionary of parameters.

        Input:
            tstart      Observation start time.
            texp        Exposure time for the observation.

        Return:         Cadence object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Metronome(tstart, texp, texp, 1)
    #===========================================================================



    #===========================================================================
    # uvt
    #===========================================================================
    def uvt(self, indices, fovmask=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return coordinates (u,v) and time t for indices into the data array.

        This method supports non-integer index values. Values exactly at the
        upper limit of indexing are treated as falling inside the field of view.

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
        slit_coord = indices.to_scalar(self.along_slit_index)

        #---------------------------------
        # Handle discontinuous detectors
        #---------------------------------
        if self.slit_is_discontinuous:

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Identify indices at exact upper limit; treat these as inside
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            at_upper_limit = (slit_coord == self.along_slit_shape)

            #- - - - - - - - - - - - - - - - - - - - - - - -
            # Map continuous index to discontinuous (u,v)
            #- - - - - - - - - - - - - - - - - - - - - - - -
            slit_int = slit_coord.int()
            slit_coord = slit_int + (slit_coord - slit_int) * self.det_size

            #- - - - - - - - - - - - - - - - -
            # Adjust values at upper limit
            #- - - - - - - - - - - - - - - - -
            slit_coord = slit_coord.mask_where(at_upper_limit,
                            replace = self.along_slit_shape + self.det_size - 1,
                            remask = False)

        #----------------------
        # Create (u,v) Pair
        #----------------------
        uv_vals = np.empty(indices.shape + (2,))
        uv_vals[..., self.along_slit_uv_axis] = slit_coord.values
        uv_vals[..., self.cross_slit_uv_axis] = 0.5
        uv = Pair(uv_vals, indices.mask)

        #-------------------------
        # Create time Scalar
        #-------------------------
        time = self.scalar_midtime

        #----------------------------
        # Apply mask if necessary
        #----------------------------
        if fovmask:
            is_outside = self.uv_is_outside(uv, inclusive=True)
            if np.any(is_outside):
                uv = uv.mask_where(is_outside)

                #- - - - - - - - - - - - -
                # Create time Scalar
                #- - - - - - - - - - - - -
                if indices.values.shape == ():
                    time_values = self.midtime
                else:
                    time_values = np.empty(indices.shape)
                    time_values[...] = self.midtime

                time = Scalar(time_values, uv.mask)

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

        slit_coord = indices.to_scalar(self.along_slit_index)

        uv_vals = np.empty(indices.shape + (2,), dtype='int')
        uv_vals[..., self.along_slit_uv_axis] = slit_coord.vals
        uv_vals[..., self.cross_slit_uv_axis] = 0
        uv_min = Pair(uv_vals, indices.mask)
        uv_max = uv_min + Pair.ONES

        time_min = self.scalar_time[0]
        time_max = self.scalar_time[1]

        if fovmask:
            is_outside = self.uv_is_outside(uv_min, inclusive=False)
            if np.any(is_outside):
                uv_min = uv_min.mask_where(is_outside)
                uv_max = uv_max.mask_where(is_outside)

                time_min_vals = np.empty(is_outside.shape)
                time_max_vals = np.empty(is_outside.shape)

                time_min_vals[...] = self.time[0]
                time_max_vals[...] = self.time[1]

                time_min = Scalar(time_min_vals, is_outside)
                time_max = Scalar(time_max_vals, is_outside)

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
            tstep       a time step index (one or two integers). Not checked for
                        out-of-range errors.

        Return:         a tuple (uv_min, uv_max)
            uv_min      a Pair defining the minimum values of (u,v) coordinates
                        active at this time step.
            uv_min      a Pair defining the maximum values of (u,v) coordinates
                        active at this time step (exclusive).
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return (Pair.ZERO, self.fov.uv_shape)
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
        if fovmask:
            is_outside = self.uv_is_outside(uv_pair, inclusive=True)
            if np.any(is_outside):
                time_min_vals = np.empty(is_outside.shape)
                time_max_vals = np.empty(is_outside.shape)

                time_min_vals[...] = self.time[0]
                time_max_vals[...] = self.time[1]

                time_min = Scalar(time_min_vals, is_outside)
                time_max = Scalar(time_max_vals, is_outside)

                return (time_min, time_max)

        return self.scalar_time

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
        return Pair.ZERO
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
#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#        obs = Slit1D(self.axes, self.det_size, 
#                    {'tstart':self.tstart + dtime, 'texp':self.texp},
#                     self.fov, self.path, self.frame)
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
        obs = Slit1D(self.axes, self.det_size, self.tstart + dtime, self.texp,
                     self.fov, self.path, self.frame)
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

        for key in self.subfields.keys():
            obs.insert_subfield(key, self.subfields[key])

        return obs
    #===========================================================================


#*******************************************************************************

################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_Slit1D
#*******************************************************************************
class Test_Slit1D(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        from oops.cadence_.metronome import Metronome
        from oops.fov_.flatfov import FlatFOV

        fov = FlatFOV((0.001,0.001), (20,1))
#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#        obs = Slit1D(axes=('u'), det_size=1, {'tstart':0., 'texp':10.},
#                   fov=fov, path='SSB', frame='J2000')
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
        obs = Slit1D(axes=('u'), det_size=1, tstart=0., texp=10.,
                   fov=fov, path='SSB', frame='J2000')
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

        indices = Vector([(0,0),(1,0),(20,0),(21,0)])

        #------------------------------------
        # uvt() with fovmask == False
        #------------------------------------
        (uv,time) = obs.uvt(indices)
        self.assertFalse(np.any(uv.mask))
        self.assertFalse(np.any(time.mask))
        self.assertEqual(time, 5.)
        self.assertEqual(uv.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv.to_scalar(1), 0.5)

        # uvt() with fovmask == True
        (uv,time) = obs.uvt(indices, fovmask=True)

        self.assertTrue(np.all(uv.mask == np.array(3*[False] + [True])))
        self.assertTrue(np.all(time.mask == uv.mask))
        self.assertEqual(time[:3], (5,5,5))
        self.assertEqual(uv[:3].to_scalar(0), indices[:3].to_scalar(0))
        self.assertEqual(uv[:3].to_scalar(1), 0.5)

        #---------------------------------------
        # uvt_range() with fovmask == False
        #---------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertFalse(np.any(uv_min.mask))
        self.assertFalse(np.any(uv_max.mask))
        self.assertFalse(np.any(time_min.mask))
        self.assertFalse(np.any(time_max.mask))

        self.assertEqual(uv_min.to_scalar(0), indices.to_scalar(0))
        self.assertEqual(uv_min.to_scalar(1), 0)
        self.assertEqual(uv_max.to_scalar(0), indices.to_scalar(0) + 1)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, 0.)
        self.assertEqual(time_max, 10.)

        #----------------------------------------
        # uvt_range() with fovmask == True
        #----------------------------------------
        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices,
                                                             fovmask=True)

        self.assertTrue(np.all(uv_min.mask == np.array(2*[False] + 2*[True])))
        self.assertTrue(np.all(uv_max.mask == uv_min.mask))
        self.assertTrue(np.all(time_min.mask == uv_min.mask))
        self.assertTrue(np.all(time_max.mask == uv_min.mask))

        self.assertEqual(uv_min.to_scalar(0)[:2], indices.to_scalar(0)[:2])
        self.assertEqual(uv_min.to_scalar(1)[:2], 0)
        self.assertEqual(uv_max.to_scalar(0)[:2], indices.to_scalar(0)[:2] + 1)
        self.assertEqual(uv_max.to_scalar(1)[:2], 1)
        self.assertEqual(time_min[:2], 0.)
        self.assertEqual(time_max[:2], 10.)

        #-----------------------------------------
        # times_at_uv() with fovmask == False
        #-----------------------------------------
        uv = Pair([(0,0),(0,0.5),(0,1),(0,2),
                   (20,0),(20,0.5),(20,1),(20,2),
                   (21,0)])

        (time0, time1) = obs.times_at_uv(uv)

        self.assertEqual(time0, 0.)
        self.assertEqual(time1, 10.)

        #---------------------------------------
        # times_at_uv() with fovmask == True
        #---------------------------------------
        (time0, time1) = obs.times_at_uv(uv, fovmask=True)

        self.assertTrue(np.all(time0.mask == 3*[False] + [True] +
                                             3*[False] + 2*[True]))
        self.assertTrue(np.all(time1.mask == time0.mask))
        self.assertEqual(time0[:3], 0.)
        self.assertEqual(time1[:3], 10.)

        ####################################


        #----------------------------------------
        # Alternative axis order ('a','u','b')
        #----------------------------------------

        fov = FlatFOV((0.001,0.001), (20,1))
#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#        obs = Slit1D(axes=('a','u', 'b'), det_size=1, {'tstart':0., 'texp':10.},
#                     fov=fov, path='SSB', frame='J2000')
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
        obs = Slit1D(axes=('a','u', 'b'), det_size=1, tstart=0., texp=10.,
                     fov=fov, path='SSB', frame='J2000')
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

        indices = Vector([(0,0,0),(0,1,99),(0,19,99),(10,20,99),(10,21,99)])

        (uv,time) = obs.uvt(indices)

        self.assertEqual(uv.to_scalar(0), indices.to_scalar(1))
        self.assertEqual(uv.to_scalar(1), 0.5)
        self.assertEqual(time, 5.)

        (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

        self.assertEqual(uv_min.to_scalar(0), indices.to_scalar(1))
        self.assertEqual(uv_max.to_scalar(0), indices.to_scalar(1)+1)
        self.assertEqual(uv_min.to_scalar(1), 0.)
        self.assertEqual(uv_max.to_scalar(1), 1)
        self.assertEqual(time_min, 0.)
        self.assertEqual(time_max, 10.)

        ####################################


        #--------------------------------------------------
        # Alternative det_size for discontinuous indices
        #--------------------------------------------------

        fov = FlatFOV((0.001,0.001), (20,1))
#+DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
#        obs = Slit1D(axes=('u'), det_size=0.8, {'tstart':0., 'texp':10.}, fov=fov,
#                     path='SSB', frame='J2000')
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

#-DEFCAD:-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
        obs = Slit1D(axes=('u'), det_size=0.8, tstart=0., texp=10., fov=fov,
                     path='SSB', frame='J2000')
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-

        eps = 1.e-14
        delta = 1.e-13
        self.assertTrue(abs(obs.uvt((6.00   ,))[0].vals[0] - 6.0) < delta)
        self.assertTrue(abs(obs.uvt((6.25   ,))[0].vals[0] - 6.2) < delta)
        self.assertTrue(abs(obs.uvt((6.50   ,))[0].vals[0] - 6.4) < delta)
        self.assertTrue(abs(obs.uvt((6.75   ,))[0].vals[0] - 6.6) < delta)
        self.assertTrue(abs(obs.uvt((7 - eps,))[0].vals[0] - 6.8) < delta)
        self.assertTrue(abs(obs.uvt((7.     ,))[0].vals[0] - 7.0) < delta)

        #-------------------------------
        # Test using scalar indices
        #-------------------------------
        below = obs.uvt((20 - eps,), fovmask=True)[0].to_scalar(0)
        exact = obs.uvt((20      ,), fovmask=True)[0].to_scalar(0)
        above = obs.uvt((20 + eps,), fovmask=True)[0].to_scalar(0)

        self.assertTrue(abs(below - 19.8) < delta)
        self.assertTrue(abs(exact - 19.8) < delta)
        self.assertTrue(abs(above.values - 20.0) < delta)
        self.assertTrue(above.mask)

        #-------------------------------
        # Test using a Vector index
        #-------------------------------
        indices = Vector([(20 - eps,), (20,), (20 + eps,)])

        u = obs.uvt(indices, fovmask=True)[0].to_scalar(0)
        self.assertTrue(abs(u[0] - 19.8) < delta)
        self.assertTrue(abs(u[1] - 19.8) < delta)
        self.assertTrue(abs(u[2].values - 20.0) < delta)
        self.assertTrue(u.mask[2])
    #===========================================================================


#*******************************************************************************

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
