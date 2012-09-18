################################################################################
# oops_/path/kepler.py: Subclasses Kepler9 and Kepler of class Path.
#
# 5/18/12 MRS - Created.
################################################################################

import numpy as np
import gravity

from oops_.path.path_ import Path, Waypoint
from oops_.array.all import *
from oops_.config import PATH_PHOTONS
from oops_.event import Event
import oops_.registry as registry
import oops_.body as body

from oops_.fittable import Fittable
from oops_.constraint import Constraint

SEMIM = 0   # elements[SEMIM] = semimajor axis (km)
MEAN0 = 1   # elements[MEAN0] = mean longitude at epoch (radians)
DMEAN = 2   # elements[DMEAN] = mean motion (radians/s)
ECCEN = 3   # elements[ECCEN] = eccentricity
PERI0 = 4   # elements[PERI0] = pericenter at epoch (radians)
DPERI = 5   # elements[DPERI] = pericenter precession rate (radians/s)
INCLI = 6   # elements[INCLI] = inclination (radians)
NODE0 = 7   # elements[NODE0] = longitude of ascending node at epoch
DNODE = 8   # elements[DNODE] = nodal regression rate (radians/s, negative)

NELEMENTS = 9

class Kepler9(Path, Fittable):
    """Subclass Kepler9 defines a fittable Keplerian orbit, which is accurate to
    first order in eccentricity and inclination. This version is always defined
    using nine orbital elements.
    """

    def __init__(self, body, epoch, elements=None, observer=None, id=None):
        """Constructor for a Kepler9 path.

        Input:
            body        a Body object defining the central planet, including its
                        gravity and its ring_frame.
            epoch       the time TDB relative to which all orbital elements are
                        defined.
            elements    a tuple, list or Numpy array containing the nine orbital
                        elements:
                a           mean radius of orbit, km.
                lon         mean longitude of orbit at epoch, radians.
                n           mean motion, radians/sec.

                e           orbital eccentricty.
                peri        longitude of pericenter at epoch, radians.
                prec        pericenter precession rate, radians/sec.

                i           inclination, radians.
                node        longitude of ascending node at epoch, radians.
                regr        nodal regression rate, radians/sec, NEGATIVE!
                        Alternatively, None to leave the object un-initialized.
            observer    an optional Path object or ID defining the observation
                        point. Used for astrometry. If provided, the path is
                        returned relative to the observer, in J2000 coordinates,
                        and with light travel time from the central planet
                        already accounted for. If None (the default), then the
                        path is defined relative to the central planet in that
                        planet's ring_frame.
            id          the name under which to register the path.
        """

        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global NELEMENTS

        self.planet = body
        self.gravity = self.planet.gravity

        if observer is None:
            self.observer = None
            self.center = self.planet.path
            self.origin_id = self.planet.path_id
            self.frame_id = body.ring_frame_id
            self.to_j2000 = Matrix3.IDENTITY
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

        self.nparams = NELEMENTS

        if id in None:
            self.path_id = registry.temporary_path_id()
        else:
            self.path_id = id

        self.cached_observation_time = None
        self.cached_planet_event = None

        self.reregister()

    ########################################

    def set_params(self, elements):
        """Part of the Fittable interface. Re-defines the path given new orbital
        elements.

        Input:
            elements    An array or list of orbital elements. In order, they are
                        [a, mean0, dmean, e, peri0, dperi, i, node0, dnode].

              a         semimajor axis (km).
              mean0     mean longitude (radians) at the epoch.
              dmean     mean motion (radians/s).
              e         eccentricity.
              peri0     longitude of pericenter (radians) at the epoch.
              dperi     pericenter precession rate (rad/s).
              i         inclination (radians).
              node0     ascending node (radians) at the epoch.
              dnode     nodal regression rate (rad/s, < 0).
        """

        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global NELEMENTS

        self.elements = np.asfarray(elements)
        assert self.elements.shape == (9,)

        # Make copies of the orbital elements for convenience
        self.a = self.elements[SEMIM]
        self.e = self.elements[ECCEN]
        self.i = self.elements[INCLI]
        self.cos_i = np.cos(self.i)
        self.sin_i = np.sin(self.i)

        self.mean0 = self.elements[MEAN0]
        self.peri0 = self.elements[PERI0]
        self.node0 = self.elements[NODE0]

        self.dmean = self.elements[DMEAN]
        self.dperi = self.elements[DPERI]
        self.dnode = self.elements[DNODE]

        self.ae = self.a * self.e

    ########################################

    def get_params(self):
        """Part of the Fittable interface. Returns the current orbital elements.
        """

        return self.elements

    ########################################

    def copy(self):
        """Part of the Fittable interface. Returns a deep copy of the object.
        """

        return Kepler9(self.planet, self.epoch, self.get_params().copy(),
                       self.observer)

    ########################################

    def get_elements(self):
        """Returns the complete set of nine orbital elements.
        """

        return self.elements

    ########################################

    def xyz_planet(self, time, partials=False):
        """Returns the body position and velocity relative to the planet as a
        function of time, in an inertial frame where the Z-axis is aligned with
        the planet's rotation pole. Optionally, it also returns the partial
        derivatives of the position vector with respect to the orbital elements,
        on the the assumption that all nine orbital elements are independent.
        The coordinates are only accurate to first order in (e,i). The
        derivatives are precise relative to the definitions of these elements.

        Input:
            time        time (seconds) as a Scalar.
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

        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global NELEMENTS

        # Convert to array if necessary
        time = Scalar.as_scalar(time)
        t = time.vals - self.epoch

        # Determine moon polar coordinates in orbit plane
        #   mean  = mean0 + dmean * t
        #   peri  = peri0 + dperi * t
        #   r     = a    * (1 - e * cos(mean - peri))
        #   theta = mean - 2 * e * sin(mean - peri)

        mean = self.mean0 + self.dmean * t
        peri = self.peri0 + self.dperi * t

        mp = mean - peri
        cos_mp = np.cos(mp)
        sin_mp = np.sin(mp)

        r     = self.a - self.ae * cos_mp
        theta = mean   - (2. * self.e * sin_mp)

        # Time-derivatives...
        d_mp_dt = (self.dmean - self.dperi)
        d_r_dt = self.ae * sin_mp * d_mp_dt
        d_theta_dt = self.dmean - 2 * self.e * cos_mp * d_mp_dt

        # Locate body on an inclined orbit, in a frame where X is along the
        # ascending node
        #   asc[X] = r cos(theta - node)
        #   asc[Y] = r sin(theta - node) cos(i)
        #   asc[Z] = r sin(theta - node) sin(i)

        node = self.node0 + self.dnode * t
        tn = theta - node
        cos_tn = np.cos(tn)
        sin_tn = np.sin(tn)

        unit1 = np.empty((time.shape + [3]))
        unit1[...,0] = cos_tn
        unit1[...,1] = sin_tn * self.cos_i
        unit1[...,2] = sin_tn * self.sin_i

        asc = r[...,np.newaxis] * unit1

        # Time-derivatives...
        d_tn_dt = d_theta_dt - self.dnode

        d_unit1_d_tn = np.empty(time.shape + [3])
        d_unit1_d_tn[...,0] = -sin_tn
        d_unit1_d_tn[...,1] =  cos_tn * self.cos_i
        d_unit1_d_tn[...,2] =  cos_tn * self.sin_i

        d_asc_dt = (d_r_dt[...,np.newaxis] * unit1 +
                    (r * d_tn_dt)[...,np.newaxis] * d_unit1_d_tn)

        # Rotate the ascending node back into position in our inertial frame
        #   xyz[X] = asc[X] * cos(node) - asc[Y] * sin(node)
        #   xyz[Y] = asc[X] * sin(node) + asc[Y] * cos(node)
        #   xyz[Z] = asc[Z]

        cos_node = np.cos(node)
        sin_node = np.sin(node)

        rotate = np.zeros(time.shape + [3,3])
        rotate[...,0,0] =  cos_node
        rotate[...,0,1] = -sin_node
        rotate[...,1,0] =  sin_node
        rotate[...,1,1] =  cos_node
        rotate[...,2,2] =  1.

        # Matrix multiply the results
        xyz = np.sum(rotate[...,:,:] * asc[...,np.newaxis,:], axis=-1)

        # Time-derivatives...
        d_rotate_d_node = np.zeros(time.shape + [3,3])
        d_rotate_d_node[...,0,0] = -sin_node
        d_rotate_d_node[...,0,1] = -cos_node
        d_rotate_d_node[...,1,0] =  cos_node
        d_rotate_d_node[...,1,1] = -sin_node

        d_rotate_dt = d_rotate_d_node * self.dnode

        d_xyz_dt = (np.sum(rotate * d_asc_dt[...,np.newaxis,:], axis=-1) +
                    np.sum(d_rotate_dt * asc[...,np.newaxis,:], axis=-1))

        # Without a derivative calculation, we're done
        result = (xyz, d_xyz_dt)
        if not partials: return result

        # Differentiate this code from above...
        # Note that we do NOT differentiate the velocities, just the positions.
        #
        # mean = self.mean0 + self.dmean * t
        # peri = self.peri0 + self.dperi * t
        # 
        # mp = mean - peri
        # 
        # r     = self.a * (1. - self.e * cos_mp)
        # theta = mean   - (2. * self.e * sin_mp)

        # Get the derivatives of r and theta with respect to each element

        d_r_d_elem = np.zeros(time.shape + [NELEMENTS])
        d_r_d_elem[...,SEMIM] = 1 - self.e  * cos_mp
        d_r_d_elem[...,ECCEN] =   - self.a  * cos_mp
        d_r_d_elem[...,MEAN0] =     self.ae * sin_mp
        d_r_d_elem[...,DMEAN] = t * d_r_d_elem[...,MEAN0]
        d_r_d_elem[...,PERI0] =   - d_r_d_elem[...,MEAN0]
        d_r_d_elem[...,DPERI] =   - d_r_d_elem[...,DMEAN]

        d_theta_d_elem = np.zeros(time.shape + [NELEMENTS])
        d_theta_d_elem[...,ECCEN] = -2 * sin_mp
        d_theta_d_elem[...,PERI0] =  2 * cos_mp * self.e
        d_theta_d_elem[...,DPERI] =  t * d_theta_d_elem[...,PERI0]
        d_theta_d_elem[...,MEAN0] =  1 - d_theta_d_elem[...,PERI0]
        d_theta_d_elem[...,DMEAN] =  t * d_theta_d_elem[...,MEAN0]

        # Differentiate this code from above...
        #
        # node = self.node0 + self.dnode * t
        # tn = theta - node
        # 
        # unit1 = np.array([cos_tn,
        #                   sin_tn * self.cos_i,
        #                   sin_tn * self.sin_i])
        # asc = r * unit1

        # Get the derivaties of the "asc" coordinates with respect to each
        # element

        d_tn_d_elem = d_theta_d_elem.copy()
        d_tn_d_elem[...,NODE0] -= 1.
        d_tn_d_elem[...,DNODE] -= t

        d_asc_d_r = unit1

        d_asc_d_tn = np.empty(time.shape + [3])
        d_asc_d_tn[...,0] = -sin_tn
        d_asc_d_tn[...,1] =  cos_tn * self.cos_i
        d_asc_d_tn[...,2] =  cos_tn * self.sin_i
        d_asc_d_tn *= r[...,np.newaxis]

        d_asc_d_i = np.zeros(time.shape + [3])
        d_asc_d_i[...,1] = -sin_tn * self.sin_i
        d_asc_d_i[...,2] =  sin_tn * self.cos_i
        d_asc_d_i *= r[...,np.newaxis]

        # Combine using outer products...
        d_asc_d_elem = (
                 d_asc_d_r[...,:,np.newaxis] *  d_r_d_elem[...,np.newaxis,:] +
                d_asc_d_tn[...,:,np.newaxis] * d_tn_d_elem[...,np.newaxis,:])

        d_asc_d_elem[...,:,INCLI] += d_asc_d_i

        # Differentiate this code from above...
        #
        # Rotate the ascending node back into position in our inertial frame
        #   xyz[X] = asc[X] * cos(node) - asc[Y] * sin(node)
        #   xyz[Y] = asc[X] * sin(node) + asc[Y] * cos(node)
        #   xyz[Z] = asc[Z]
        # 
        # xyz = rotate <dot> asc

        # Get the derivaties of the "xyz" coordinates with respect to each
        # element

        d_node_d_elem = np.zeros(time.shape + [NELEMENTS])
        d_node_d_elem[...,NODE0] = 1.
        d_node_d_elem[...,DNODE] = t

        d_rotate_d_elem = (d_rotate_d_node[...,:,:,np.newaxis] *
                           d_node_d_elem[...,np.newaxis,np.newaxis,:])

        d_xyz_d_elem = (np.sum(rotate[...,:,:,np.newaxis] *
                               d_asc_d_elem[...,np.newaxis,:,:], axis=-2) +
                        np.sum(d_rotate_d_elem[...,:,:,:] *
                               asc[...,np.newaxis,:,np.newaxis], axis=-2))

        result += (d_xyz_d_elem,)
        return result

    ########################################

    def xyz_observed(self, time, planet_event, partials=False):
        """Returns the body position and velocity relative to the observer and
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

        xyz_planet = self.xyz_planet(planet_event.time, partials)

        pos_j2000 = self.to_j2000.rotate(xyz_planet[0]) + planet_event.pos
        vel_j2000 = self.to_j2000.rotate(xyz_planet[1]) + planet_event.vel

        if not partials: return (pos_j2000, vel_j2000)

        # Rotate derivatives to J2000
        d_xyz_d_elem_vals = xyz_planet[2]                   # shape (...,3,9)

        derivs = Vector3(d_xyz_d_elem_vals.swapaxes(-1,-2)) # shape [...,9]

        derivs_j2000 = self.to_j2000.rotate(derivs)         # shape [...,9]

        derivs_j2000_vals = derivs_j2000.vals               # shape (...,9,3)

        dpos_delem_j2000 = MatrixN(derivs_j2000_vals.swapaxes(-1,-2))
                                                            # shape [...]
                                                            # item shape [3,9]

        return (pos_j2000, vel_j2000, dpos_delem_j2000)

    ########################################

    def event_at_time(self, time, quick=None, partials=False):
        """Returns an Event object corresponding to a specified Scalar time on
        this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity of the path.
        """

        if self.observer is None:
            (pos, vel) = self.xyz_planet(time)
            return Event(time, pos, vel, self.origin_id, self.frame_id)

        if time == self.cached_observation_time:
            planet_event = self.cached_planet_event
        else:
            observer_event = Event(time, Vector3.ZERO, Vector3.ZERO,
                                   self.observer.path_id, self.frame_id)
            planet_event = self.center.photon_to_event(observer_event,
                                                       quick=quick)
            self.cached_observation_time = time
            self.cached_planet_event = planet_event

        xyz_observed = self.xyz_observed(time, planet_event, partials)

        event = Event(time, xyz_observed[0], xyz_observed[1],
                            self.observer.path_id, self.frame_id)

        if partials:
            event.pos.insert_subfield("d_dpath", xyz_observed[2])

        return event

    ####################################################
    # Override for the case where observer != None
    ####################################################

    def photon_to_event(self, link, quick=None, derivs=False, guess=None,
                              update=True,
                              iters     = PATH_PHOTONS.max_iterations,
                              precision = PATH_PHOTONS.dlt_precision,
                              limit     = PATH_PHOTONS.dlt_limit,
                              partials=False):
        """Returns the departure event at the given path for a photon, given the
        linking event of the photon's arrival. See _solve_photon() for details.
        """

        if self.observation is None:
            return super(Kepler9,self).photon_to_event(link, quick, derivs,
                                                       guess, update,
                                                       iters, precision, limit)

        # When the observer is pre-defined, photon_to_event() gets an override
        # for quick evaluation, but is only valid at the observer event.
        event = self.event_at_time(time, quick=quick, partials=partials)

        light_time = pos.norm() / oops.C
        event.time = event.time - light_time
        event.dep = -event.pos
        event.dep_lt = light_time
        event.link = link

        link.arr = event.dep
        link.arr_lt = -event.dep_lt

        if partials:
            link.arr.insert_subfield("d_dpath", -event.pos.d_dpath)
            link.arr.insert_subfield("d_dt", -event.vel)

        return event

################################################################################
# Class Kepler
################################################################################

def FUNC_A_FROM_N(obj, n, derivs=False):
    a = obj.gravity.solve_a(n, (1,0,0))
    if not derivs: return a
    return (a, 1. / obj.gravity.d_dmean_dt_da(a))

def FUNC_N_FROM_A(obj, a, derivs=False):
    n = obj.gravity.n(a)
    if not derivs: return n
    return (n, obj.gravity.d_dmean_dt_da(a))

def FUNC_DPERI_FROM_A(obj, a, derivs=False):
    dperi = obj.gravity.dperi_dt(a)
    if not derivs: return dperi
    return (dperi, obj.gravity.d_dperi_dt_da(a))

def FUNC_DPERI_FROM_N(obj, n, derivs=False):
    a = obj.gravity.solve_a(n, (1,0,0))
    dperi = obj.gravity.dperi_dt(a)
    if not derivs: return dperi
    return (dperi, obj.gravity.d_dperi_dt_da(a) / obj.gravity.d_dmean_dt_da(a))

def FUNC_DNODE_FROM_A(obj, a, derivs=False):
    dnode = obj.gravity.dnode_dt(a)
    if not derivs: return dnode
    return (dnode, obj.gravity.d_dnode_dt_da(a))

def FUNC_DNODE_FROM_N(obj, n, derivs=False):
    a = obj.gravity.solve_a(n, (1,0,0))
    dnode = obj.gravity.dnode_dt(a)
    if not derivs: return dnode
    return (dnode, obj.gravity.d_dnode_dt_da(a) / obj.gravity.d_dmean_dt_da(a))

class Kepler(Kepler9, Fittable):
    """Subclass Kepler defines a fittable Keplerian orbit, which is accurate to
    first order in eccentricity and inclination. This is similar to subclass
    Kepler9 but allows for coupled orbital elements.
    """

    def __init__(self, body, epoch, elements, observer=None, id=None):
        """Constructor for a Kepler path.

        Input:
            body        a Body object defining the central planet, including its
                        gravity and its ring_frame.
            epoch       the time TDB relative to which all orbital elements are
                        defined.
            elements    a tuple, list or Numpy array containing the nine orbital
                        elements:
                a           mean radius of orbit, km; "N" to derive it from n.
                lon         mean longitude of orbit at epoch, radians.
                n           mean motion, radians/sec; "A" to derive it from a.

                e           orbital eccentricity.
                peri        longitude of pericenter at epoch, radians.
                prec        pericenter precession rate, radians/sec; "A" or "N"
                            to derive it from a or n.

                i           inclination, radians.
                node        longitude of ascending node at epoch, radians.
                regr        nodal regression rate, radians/sec, NEGATIVE!; "A"
                            or "N" to derive it from a or n.

            observer    an optional Path object or ID defining the observation
                        point. Used for astrometry. If provided, the path is
                        returned relative to the observer, in J2000 coordinates,
                        and with light travel time from the central planet
                        already accounted for. If None (the default), then the
                        path is defined relative to the central planet in that
                        planet's ringframe.
            id          the name under which to register the path.
        """

        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global NELEMENTS

        assert ("A" not in elements) or ("N" not in elements)
        assert (elements[SEMIM] != "N") or (elements[DMEAN] != "A")

        # Create an un-initialized Kepler9 object
        if id is None:
            self.path_id = registry.temporary_path_id()
        else:
            self.path_id = id

        self.kepler9 = Kepler9(body, epoch, None, observer,
                               self.path_id + "_KEPLER9")

        # Interpret the constraints
        constraints = NELEMENTS * [None]

        if elements[SEMIM] == "N":
            constraints[SEMIM] = (FUNC_A_FROM_N, DMEAN)

        if elements[DMEAN] == "A":
            constraints[DMEAN] = (FUNC_N_FROM_A, SEMIM)

        if elements[DPERI] == "A":
            constraints[DPERI] = (FUNC_DPERI_FROM_A, SEMIM)
        elif elements[DPERI] == "N":
            constraints[DPERI] = (FUNC_DPERI_FROM_N, DMEAN)

        if elements[DNODE] == "A":
            constraints[DNODE] = (FUNC_DNODE_FROM_A, SEMIM)
        elif elements[DNODE] == "N":
            constraints[DNODE] = (FUNC_DNODE_FROM_N, DMEAN)

        self.initial_elements = elements
        self.constraint = Constraint(self.kepler9, constraints)
        self.params = self.constraint.new_params(elements)
        self.nparams = self.constraint.nparams
        self.kepler9.set_params(self.constraint.old_params(self.params))

        # Complete and register
        self.gravity = body.gravity

        self.origin_id = self.kepler9.origin_id
        self.frame_id = "J2000"
        self.reregister()

    ########################################

    def unregister(self):
        """Override of the standard Path method."""

        self.kepler9.unregister()
        super(Kepler,self).unregister()

    ########################################

    def set_params(self, params):
        """Part of the Fittable interface. Re-defines the path given new
        parameters."""

        self.params = params
        self.kepler9.set_params(self.constraint.old_params(params))

    ########################################

    def get_params(self):
        """Part of the Fittable interface. Re-defines the path given new
        parameters."""

        return self.params

    ########################################

    def copy(self):
        """Part of the Fittable interface. Returns a deep copy of the object.
        """

        kep = Kepler(self.kepler9.planet, self.kepler9.epoch,
                     self.initial_elements, self.kepler9.observer)
        kep.set_params(self.get_params())
        return kep

    ########################################

    def get_elements(self):
        """Returns the complete set of nine orbital elements.
        """

        return self.kepler9.get_elements()

    ########################################

    def event_at_time(self, time, quick=None,
                            planet_event=None, partials=False):
        """Returns an Event object corresponding to a specified Scalar time on
        this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity of the path.
        """

        event = self.kepler9.event_at_time(time, quick, planet_event, partials)

        # Update the partials if necessary
        if partials:
            dpos_dpath = self.constraint.partials_wrt_new(self.params,
                                                    event.pos.d_dpath.vals)
            event.pos.insert_subfield("d_dpath", MatrixN(dpos_dpath))

        return event

################################################################################
# UNIT TESTS
################################################################################

import unittest

def _xyz_planet_derivative_test(kep, t, delta=1.e-6):
    """Calculates numerical derivatives of (x,y,z) in the planet frame relative
    to the orbital elements, at time(s) t. It returns a tuple of (numerical
    derivatives, analytic derivatives, relative errors). Used for debugging.
    """

    # Save the position and its derivatives
    (xyz, d_xyz_dt, d_xyz_d_elem) = kep.xyz_planet(t, partials=True)
    pos_norm = np.sqrt(np.sum(xyz**2, axis=-1))
    vel_norm = np.sqrt(np.sum(d_xyz_dt**2, axis=-1))

    # Create new Kepler objects for tweaking the parameters
    khi = kep.copy()
    klo = kep.copy()

    params = kep.get_params()

    # Loop through parameters...
    new_derivs = np.zeros(np.shape(t) + (3,kep.nparams))
    errors = np.zeros(np.shape(t) + (3,kep.nparams))
    for e in range(kep.nparams):

        # Tweak one parameter
        hi = params.copy()
        lo = params.copy()

        if params[e] == 0.:
            hi[e] =  delta
            lo[e] = -delta
            denom = 2. * delta
        else:
            hi[e] *= 1. + delta
            lo[e] *= 1. - delta
            denom = hi[e] - lo[e]

        khi.set_params(hi)
        klo.set_params(lo)

        # Determine the partial derivative
        xyz_hi = khi.xyz_planet(t, partials=False)[0]
        xyz_lo = klo.xyz_planet(t, partials=False)[0]
        xyz_diff = xyz_hi - xyz_lo
        new_derivs[...,:,e] = xyz_diff / denom

        # Test slopes and test against numeric precision
        norm = np.maximum(np.abs(new_derivs[...,:,e]),
                          np.abs(d_xyz_d_elem[...,:,e]))
        norm[norm == 0.] = 1.
        errors[...,:,e] = (np.abs(new_derivs[...,:,e] -
                           d_xyz_d_elem[...,:,e])) / norm

        precision = (np.abs(xyz_diff) /
                     np.maximum(pos_norm,
                                vel_norm * np.abs(t))[..., np.newaxis])
        precision[precision == 0.] = 1.
        errors[...,:,e] = np.minimum(errors[...,:,e],
                                     1.e-16/precision * norm)

    klo.unregister()
    khi.unregister()

    return (new_derivs, d_xyz_d_elem, errors)

def _pos_derivative_test(kep, t, delta=1.e-6):
    """Calculates numerical derivatives of (x,y,z) in the observer/J2000 frame
    relative to the orbital elements, at time(s) t. It returns a tuple of
    (numerical derivatives, analytic derivatives, relative errors). Used for
    debugging.
    """

    # Save the position and its derivatives
    event = kep.event_at_time(t, partials=True)
    xyz = event.pos.vals
    d_xyz_dt = event.vel.vals
    d_xyz_d_elem = event.pos.d_dpath.vals

    pos_norm = np.sqrt(np.sum(xyz**2, axis=-1))
    vel_norm = np.sqrt(np.sum(d_xyz_dt**2, axis=-1))

    # Create new Kepler objects for tweaking the parameters
    khi = kep.copy()
    klo = kep.copy()

    params = kep.get_params()

    # Loop through parameters...
    new_derivs = np.zeros(np.shape(t) + (3,kep.nparams))
    errors = np.zeros(np.shape(t) + (3,kep.nparams))
    for e in range(kep.nparams):

        # Tweak one parameter
        hi = params.copy()
        lo = params.copy()

        if params[e] == 0.:
            hi[e] =  delta
            lo[e] = -delta
            denom = 2. * delta
        else:
            hi[e] *= 1. + delta
            lo[e] *= 1. - delta
            denom = hi[e] - lo[e]

        khi.set_params(hi)
        klo.set_params(lo)

        # Determine the partial derivative
        xyz_hi = khi.event_at_time(t, partials=False).pos.vals
        xyz_lo = klo.event_at_time(t, partials=False).pos.vals
        xyz_diff = xyz_hi - xyz_lo
        new_derivs[...,:,e] = xyz_diff / denom

        # Test slopes and test against numeric precision
        norm = np.maximum(np.abs(new_derivs[...,:,e]),
                          np.abs(d_xyz_d_elem[...,:,e]))
        norm[norm == 0.] = 1.
        errors[...,:,e] = (np.abs(new_derivs[...,:,e] -
                           d_xyz_d_elem[...,:,e])) / norm

        precision = (np.abs(xyz_diff) /
                     np.maximum(pos_norm,
                                vel_norm * np.abs(t))[..., np.newaxis])
        precision[precision == 0.] = 1.
        errors[...,:,e] = np.minimum(errors[...,:,e],
                                     1.e-16/precision * norm)

    klo.unregister()
    khi.unregister()

    return (new_derivs, d_xyz_d_elem, errors)

class Test_Kepler(unittest.TestCase):

    def runTest(self):
        # SEMIM = 0    elements[SEMIM] = semimajor axis (km)
        # MEAN0 = 1    elements[MEAN0] = mean longitude at epoch (radians)
        # DMEAN = 2    elements[DMEAN] = mean motion (radians/s)
        # ECCEN = 3    elements[ECCEN] = eccentricity
        # PERI0 = 4    elements[PERI0] = pericenter at epoch (radians)
        # DPERI = 5    elements[DPERI] = pericenter precession rate (radians/s)
        # INCLI = 6    elements[INCLI] = inclination (radians)
        # NODE0 = 7    elements[NODE0] = longitude of ascending node at epoch
        # DNODE = 8    elements[DNODE] = nodal regression rate (radians/s)

        body.define_solar_system("1999-01-01", "2002-01-01")

        a = 140000.

        saturn = gravity.SATURN
        dmean = saturn.n(a)
        dperi = saturn.dperi_dt(a)
        dnode = saturn.dnode_dt(a)

        TIMESTEPS = 500
        time = 3600. * np.arange(TIMESTEPS)

        kep9 = Kepler9(registry.body_lookup("SATURN"), 0.,
                       (a, 1., dmean, 0.2, 2., dperi, 4., 0., dnode),
                       registry.path_lookup("EARTH"))

        errors = _xyz_planet_derivative_test(kep9, time)[2]
        self.assertTrue(np.all(np.abs(errors) < 1.e-4))

        errors = _pos_derivative_test(kep9, time)[2]
        self.assertTrue(np.all(np.abs(errors) < 1.e-4))

        kep = Kepler(registry.body_lookup("SATURN"), 0.,
                     (a, 1., "A", 0.2, 2., "A", 4., 0., "A"),
                     registry.path_lookup("EARTH"))
        self.assertEqual(kep.nparams, 6)
        self.assertEqual(kep.kepler9.nparams, 9)
        self.assertEqual(kep.constraint.fittable, kep.kepler9)
        self.assertEqual(kep.constraint.nparams, kep.nparams)

        errors = _pos_derivative_test(kep, time)[2]
        self.assertTrue(np.all(np.abs(errors) < 1.e-4))

        kep = Kepler(registry.body_lookup("SATURN"), 0.,
                      ("N", 1., dmean, 0.2, 2., "N", 4., 0., "N"),
                     registry.path_lookup("EARTH"))
        self.assertEqual(kep.nparams, 6)
        self.assertEqual(kep.kepler9.nparams, 9)
        self.assertEqual(kep.constraint.fittable, kep.kepler9)
        self.assertEqual(kep.constraint.nparams, kep.nparams)

        errors = _pos_derivative_test(kep, time)[2]
        self.assertTrue(np.all(np.abs(errors) < 1.e-4))

        registry.initialize_frame_registry()
        registry.initialize_path_registry()
        registry.initialize_body_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
