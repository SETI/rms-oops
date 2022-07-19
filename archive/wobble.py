################################################################################
# oops/path_/wobble.py: Subclass Wobble of class Path.
#
# 10/30/12 MRS - Defined. NOTE: For now, partial derivatives do not work. Also,
#                the wobble is in the (x,y) plane, neglecting any orbital
#                inclination.
# 11/1/12 MRS - Revised to handle multiple independent wobbles
################################################################################

import numpy as np
from polymath import *
import gravity

from oops.path_.path   import Path, Waypoint
from oops.path_.kepler import Kepler
from oops.config       import PATH_PHOTONS
from oops.event        import Event
from oops.fittable     import Fittable

import oops.registry  as registry
import oops.constants as constants

SEMIM = 0   # elements[SEMIM] = semimajor axis (km).
MEAN0 = 1   # elements[MEAN0] = mean longitude at epoch (radians).
DMEAN = 2   # elements[DMEAN] = mean motion (radians/s).
ECCEN = 3   # elements[ECCEN] = eccentricity.
PERI0 = 4   # elements[PERI0] = pericenter at epoch (radians).
DPERI = 5   # elements[DPERI] = pericenter precession rate (radians/s).
INCLI = 6   # elements[INCLI] = inclination (radians).
NODE0 = 7   # elements[NODE0] = longitude of ascending node at epoch.
DNODE = 8   # elements[DNODE] = nodal regression rate (radians/s, negative).
RWOBB = 9   # elements[RWOBB] = radial amplitude of wobble.
RATIO = 10  # elements[RATIO] = ratio of longitude amplitude to radial
            #                   amplitude.
WOBB0 = 11  # elements[WOBB0] = wobble longitude at epoch, measured from the
            #                   radial minimum of the epicycle.
DWOBB = 12  # elements[DWOBB] = rate of wobble epicyclic rotation (radians/s).

KEPLER_ELEMENTS = 9
WOBBLE_ELEMENTS = 4

#*******************************************************************************
# Wobble
#*******************************************************************************
class Wobble(Path, Fittable):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Subclass Wobble defines a fittable Keplerian orbit, with an additional
    component of "wobble" or libration.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, body, epoch, wobbles=1, elements=None, observer=None,
                       id=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Wobble path.

        Input:
            body        a Body object defining the central planet, including its
                        gravity and its ring_frame.
            epoch       the time TDB relative to which all orbital elements are
                        defined.
            wobbles     the number of wobble epicycles to superimpose upon the
                        orbital motion.
            elements    a tuple, list or Numpy array containing the nine Kepler
                        orbital elements plus four more for each wobble.
                        Alternatively, set to None in order to leave the set of
                        elements un-initialized.
                a           mean radius of orbit, km.
                lon         mean longitude of orbit at epoch, radians.
                n           mean motion, radians/sec.

                e           orbital eccentricity.
                peri        longitude of pericenter at epoch, radians.
                prec        pericenter precession rate, radians/sec.

                i           inclination, radians.
                node        longitude of ascending node at epoch, radians.
                regr        nodal regression rate, radians/sec, NEGATIVE!

                rwobble     radial amplitude of wobble epicycle, km.
                ratio       the ratio of the longitude amplitude to the radial
                            amplitude; typically two for standard eccentric
                            motion.
                wobble0     longitude along wobble epicycle at epoch, radians.
                            A value of zero means that this wobble is going
                            through its radial minimum point at epoch.
                dwobble_dt  rate of rotation about the wobble epicycle,
                            radians/s. Positive for prograde wobble at radial
                            minimum.
            observer    an optional Path object or ID defining the observation
                        point. Used for astrometry. If provided, the path is
                        returned relative to the observer, in J2000 coordinates,
                        and with light travel time from the central planet
                        already accounted for. If None (the default), then the
                        path is defined relative to the central planet in that
                        planet's ring_frame.
            id          the name under which to register the path.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global RWOBB, RATIO, WOBB0, DWOBB
        global KEPLER_ELEMENTS, WOBBLE_ELEMENTS

        self.nwobbles = wobbles
        self.nparams = KEPLER_ELEMENTS + WOBBLE_ELEMENTS * self.nwobbles
        self.param_name = "elements"
        self.cache = {}

        self.path_id = id or registry.temporary_path_id()

        self.kepler = Kepler(body, epoch, elements[:9], observer,
                             self.path_id + "_KEPLER")

        self.planet = self.kepler.planet
        self.origin_id = self.kepler.path_id
        self.gravity = self.kepler.gravity

        observer = None     # FOR NOW; ALTERNATIVE DOES NOT WORK
        if observer is None:
            self.observer = None
            self.center = self.planet.path
            self.origin_id = self.planet.path_id
            self.frame_id = body.ring_frame_id
            self.to_j2000 = Matrix3.UNIT
        else:
            self.observer = registry.as_path(observer)
            assert self.observer.shape == []
            self.center = Path.connect(self.planet.path_id, observer, "J2000")
            self.frame_id = "J2000"
            frame = registry.connect_frames("J2000", body.ring_frame_id)
            self.to_j2000 = frame.transform_at_time(epoch).matrix

        self.epoch = epoch

        if elements is None:
            self.elements = None
        else:
            self.set_params(elements)

        self.reregister()
    #========================================================================



    #===========================================================================
    # set_params_new
    #===========================================================================
    def set_params_new(self, elements):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Part of the Fittable interface. Re-defines the path given new orbital
        elements.

        Input:
            elements    An array or list of orbital elements. In order, they are
                        [a, mean0, dmean, e, peri0, dperi, i, node0, dnode],
                        followed by [rwobble, ratio, wobble0, dwobble_dt] for
                        each wobble.

              a         semimajor axis (km).
              mean0     mean longitude (radians) at the epoch.
              dmean     mean motion (radians/s).
              e         eccentricity.
              peri0     longitude of pericenter (radians) at the epoch.
              dperi     pericenter precession rate (rad/s).
              i         inclination (radians).
              node0     ascending node (radians) at the epoch.
              dnode     nodal regression rate (rad/s, < 0).

              rwobble   radial amplitude of wobble epicycle, km.
              ratio     the ratio of the longitude amplitude to the radial
                        amplitude; typically two for standard eccentric
                        motion.
              wobble0   longitude along wobble epicycle at epoch, radians.
              dwobble_dt  rate of rotation about the wobble epicycle, radians/s.
                        Positive for prograde wobble at radial minimum.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global RWOBB, RATIO, WOBB0, DWOBB
        global KEPLER_ELEMENTS, WOBBLE_ELEMENTS

        self.elements = np.asfarray(elements)
        assert self.elements.shape == (self.nparams,)

        self.kepler.set_params(elements[:9])

        for i in range(self.nwobbles):
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Make copies of the orbital elements for convenience
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
            self.r_wobble   = self.elements[RWOBB::WOBBLE_ELEMENTS]
            self.l_wobble   = self.elements[RATIO::WOBBLE_ELEMENTS]
            self.l_wobble  *= self.r_wobble

            self.wobble0    = self.elements[WOBB0::WOBBLE_ELEMENTS]
            self.dwobble_dt = self.elements[DWOBB::WOBBLE_ELEMENTS]

        self.rv_wobble = self.r_wobble * self.dwobble_dt
        self.lv_wobble = self.l_wobble * self.dwobble_dt

        #--------------------------------------------------------------------
        # Empty the cache
        #--------------------------------------------------------------------
        self.cached_observation_time = None
        self.cached_planet_event = None
    #========================================================================



    #===========================================================================
    # copy
    #===========================================================================
    def copy(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Part of the Fittable interface. Returns a deep copy of the object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Wobble(self.planet, self.epoch, self.get_params().copy(),
                      self.observer)
    #========================================================================



    #===========================================================================
    # get_elements
    #===========================================================================
    def get_elements(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Returns the complete set of nine orbital elements.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.elements
    #========================================================================



    #===========================================================================
    # xyz_planet
    #===========================================================================
    def xyz_planet(self, time, partials=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Returns the body position and velocity relative to the planet as a
        function of time, in an inertial frame where the Z-axis is aligned with
        the planet's rotation pole. Optionally, it also returns the partial
        derivatives of the position vector with respect to the orbital elements,
        on the the assumption that all nine orbital elements are independent.
        The coordinates are only accurate to first order in (e,i).

        Partial derivatives are not supported.

        Input:
            time        time (seconds) as a Scalar.
            partials    False (default) to return the position and velocity;
                        True to return partial derivatives as well.

        Return:
            xyz         a Vector3 of position vectors.
            dxyz_dt     a Vector3 of velocity vectors.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert not partials
        (xyz, dxyz_dt) = self.kepler.xyz_planet(time, partials=False)

        #--------------------------------------------------------------------
        # Convert time to an array if necessary
        #--------------------------------------------------------------------
        time = Scalar.as_scalar(time)
        t = time.vals - self.epoch

        #--------------------------------------------------------------------
        # Add the offsets in position and velocity in the wobble epicycles
        # The last axis in all arrays holds the wobble index
        #--------------------------------------------------------------------
        wlon = self.wobble0 + self.dwobble_dt * t[..., np.newaxis]
        cos_wlon = np.cos(wlon)
        sin_wlon = np.sin(wlon)

        r = -cos_wlon * self.r_wobble
        l =  sin_wlon * self.l_wobble
        dr_dt = sin_wlon * self.rv_wobble
        dl_dt = cos_wlon * self.lv_wobble

        #--------------------------------------------------------------------
        # Rotate into the xyz frame
        #--------------------------------------------------------------------
        mean = self.kepler.mean0 + self.kepler.dmean * t[..., np.newaxis]
        cos_mean = np.cos(mean)
        sin_mean = np.sin(mean)

        #--------------------------------------------------------------------
        # NOTE: This does not accommodate inclination. This is OK because we do
        # not claim accuracy to order (e*i).
        #--------------------------------------------------------------------
        xyz.vals[...,0] += np.sum(r*cos_mean - l*sin_mean, axis=-1)
        xyz.vals[...,1] += np.sum(r*sin_mean + l*cos_mean, axis=-1)

        dxyz_dt.vals[...,0] += np.sum(dr_dt*cos_mean - dl_dt*sin_mean, axis=-1)
        dxyz_dt.vals[...,1] += np.sum(dr_dt*sin_mean + dl_dt*cos_mean, axis=-1)

        return (xyz, dxyz_dt)
    #========================================================================



    #===========================================================================
    # xyz_observed
    #===========================================================================
    def xyz_observed(self, time, planet_event, partials=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Returns the body position and velocity relative to the observer and
        in the J2000 coordinate frame. Optionally, it also returns the partial
        derivatives of the position with respect to the orbital elements,
        assuming they are all independent.

        Input:
            time        time (seconds) as a Scalar.
            planet_event
                        the corresponding event of the photon leaving the
                        planet; None to calculate this from the time. Note that
                        this can be calculated once using planet_event(),
                        avoiding the need to re-calculate this quantity for
                        repeated calls using the same time(s).
            partials    False (default) to return the position and velocity;
                        True to return partial derivatives as well.

        Return:
            xyz         a Vector3 of position vectors.
            dxyz_dt     a Vector3 of velocity vectors.
            dxyz_delements
                        a MatrixN containing the partial derivatives of the
                        position vectors with respect to each orbital element,
                            d(x,y,z)/d(elements)
                        The item shape is [3,9].

            If partials is False, then the tuple (xyz, dxyz_dt) is returned;
            Otherwise, the tuple (xyz, dxyz_dt, dxyz_delements) is returned.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        xyz_planet = self.xyz_planet(planet_event.time, partials)

        pos_j2000 = self.to_j2000.rotate(xyz_planet[0]) + planet_event.pos
        vel_j2000 = self.to_j2000.rotate(xyz_planet[1]) + planet_event.vel

        if not partials: return (pos_j2000, vel_j2000)

        #--------------------------------------------------------------------
        # Rotate derivatives to J2000
        #--------------------------------------------------------------------
        d_xyz_d_elem_vals = xyz_planet[2]                   # shape (...,3,9)

        derivs = Vector3(d_xyz_d_elem_vals.swapaxes(-1,-2)) # shape [...,9]

        derivs_j2000 = self.to_j2000.rotate(derivs)         # shape [...,9]

        derivs_j2000_vals = derivs_j2000.vals               # shape (...,9,3)

        dpos_delem_j2000 = MatrixN(derivs_j2000_vals.swapaxes(-1,-2))
                                                            # shape [...]
                                                            # item shape [3,9]

        return (pos_j2000, vel_j2000, dpos_delem_j2000)
    #========================================================================



    #===========================================================================
    # event_at_time
    #===========================================================================
    def event_at_time(self, time, quick=None, partials=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Returns an Event object corresponding to a specified Scalar time on
        this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity of the path.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.observer is None:
            (pos, vel) = self.xyz_planet(time, partials=partials)
            return Event(time, pos, vel, self.origin_id, self.frame_id)

            observer_event = Event(time, Vector3.ZERO, Vector3.ZERO,
                                   self.observer.path_id, self.frame_id)
            planet_event = self.center.photon_to_event(observer_event,
                                                       quick=quick)

        xyz_observed = self.xyz_observed(time, planet_event, partials)

        event = Event(time, xyz_observed[0], xyz_observed[1],
                            self.observer.path_id, self.frame_id)

        if partials: 
            event.pos.insert_subfield("d_dpath", xyz_observed[2])

        return event
    #========================================================================



    ####################################################
    # Override for the case where observer != None
    ####################################################

#     #===========================================================================
#     # photon_to_event
#     #===========================================================================
#     def photon_to_event(self, link, quick=None, derivs=False, guess=None,
#                               update=True,
#                               iters     = PATH_PHOTONS.max_iterations,
#                               precision = PATH_PHOTONS.dlt_precision,
#                               limit     = PATH_PHOTONS.dlt_limit,
#                               partials=False):
#         #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         """Returns the departure event at the given path for a photon, given the
#         linking event of the photon's arrival. See _solve_photon() for details.
#         """
#         #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         if self.observer is None or guess is None:
#             return super(Kepler,self).photon_to_event(link, quick, derivs,
#                                                       guess, update,
#                                                       iters, precision, limit)
# 
#         #---------------------------------------------------------------------
#         # When the observer is pre-defined, photon_to_event() gets an override
#         # for quick evaluation, but is only valid at the observer event.
#         # In this case, guess must contain the event time at the planet.
#         #---------------------------------------------------------------------
#         event = self.event_at_time(guess, quick=quick, partials=partials)
# 
#         event.dep = -event.pos
#         event.dep_lt = event.time - link.time
#         event.link = link
# 
#         link.arr = event.dep
#         link.arr_lt = -event.dep_lt
# 
#         if partials:
#             link.arr.insert_subfield("d_dpath", -event.pos.d_dpath)
#             link.arr.insert_subfield("d_dt", -event.vel)
# 
#         return event
#     #========================================================================


#*******************************************************************************


################################################################################
# UNIT TESTS
################################################################################

#*******************************************************************************
# Test_Wobble
*******************************************************************************
# class Test_Wobble(unittest.TestCase):
# 
#     #=========================================================================
#     # runTest
#     #========================================================================
#     def runTest(self):
# 
#         # TBD
#         pass
#     #========================================================================


#*******************************************************************************

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
