################################################################################
# oops/path_/kepler.py: Subclass Kepler of class Path.
################################################################################

import numpy as np
from polymath import *

from oops.event            import Event
from oops.body             import Body
from oops.path_.path       import Path
from oops.frame_.frame     import Frame
from oops.gravity_.gravity import Gravity
from oops.fittable         import Fittable
from oops.config           import PATH_PHOTONS
import oops.constants  as constants

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

LIBAMP = 0  # libration[LIBAMP] = libration amplitude (radians)
PHASE0 = 1  # libration[PHASE0] = libration phase at epoch (radians)
DPHASE = 2  # libration[DPHASE] = libration rate (radians/s)

NWOBBLES = 3

#*******************************************************************************
# Kepler
#*******************************************************************************
class Kepler(Path, Fittable):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Subclass Kepler defines a fittable Keplerian orbit, which is accurate to
    first order in eccentricity and inclination. It is defined using nine
    orbital elements.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['planet', 'epoch', 'elements', 'observer', 'wobbles',
                    'frame', 'path_id']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, body, epoch, elements=None, observer=None, wobbles=(),
                       frame=None, id=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Kepler path.

        Input:
            body        a Body object defining the central planet, including its
                        gravity and its ring_frame.
            epoch       the time TDB relative to which all orbital elements are
                        defined.

            elements    a tuple, list or Numpy array containing the orbital
                        elements and wobble terms:
                a           mean radius of orbit, km.
                lon         mean longitude of orbit at epoch, radians.
                n           mean motion, radians/sec.

                e           orbital eccentricity.
                peri        longitude of pericenter at epoch, radians.
                prec        pericenter precession rate, radians/sec.

                i           inclination, radians.
                node        longitude of ascending node at epoch, radians.
                regr        nodal regression rate, radians/sec, NEGATIVE!

                Repeat for each wobble:

                amp         amplitude of the first wobble term, radians.
                phase0      initial phase of the first wobble term, radians.
                dphase_dt   rate of change of the first wobble term, radians/s.

                        Alternatively, a dictionary containing keys with these
                        names, or None to leave the object un-initialized.

            observer    an optional Path object or ID defining the observation
                        point. Used for astrometry. If provided, the path is
                        returned relative to the observer, in J2000 coordinates,
                        and with light travel time from the central planet
                        already accounted for. If None (the default), then the
                        path is defined relative to the central planet in that
                        planet's ring_frame.

            wobbles     a string or tuple of strings containing the name of each
                        element to which the corresponding wobble applies. Use
                        'mean', 'peri' or 'node', 'a', 'e', or 'i', for
                        individual elements. Use 'e2d' for a forced eccentricity
                        and 'i2d' for a forced inclination. Use 'pole' for a
                        Laplace plane offset).

            frame       an optional frame in which the orbit is defined. By
                        default, this is the ring_frame of the planet. Ignored
                        if observer is defined.

            id          the name under which to register the path.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global NELEMENTS

        global LIBAMP, PHASE0, DPHASE
        global NWOBBLES

        if type(wobbles) == str:
            self.wobbles = (wobbles,)
        else:
            self.wobbles = wobbles

        self.nwobbles = len(wobbles)
        for name in self.wobbles:
            assert name in ('mean', 'peri', 'node', 'a', 'e', 'i', 'e2d', 'i2d',
                            'pole')

        self.nparams = NELEMENTS + self.nwobbles * NWOBBLES
        self.param_name = "elements"
        self.cache = {}

        self.planet = Body.as_body(body)
        self.center = self.planet.path
        self.gravity = self.planet.gravity

        if observer is None:
            self.observer = None
            self.origin = self.planet.path
            self.frame  = frame or self.planet.ring_frame
            self.to_j2000 = Matrix3.IDENTITY
        else:
            self.observer = Path.as_path(observer)
            assert self.observer.shape == ()
            self.origin = self.observer
            self.frame = frame or Frame.J2000
            frame = self.frame.wrt(self.planet.ring_frame)
            self.to_j2000 = frame.transform_at_time(epoch).matrix

        self.epoch = epoch

        if elements is None:
            self.elements = None
        elif isinstance(elements, dict):
            items = [
                elements['a'],
                elements['mean0'],
                elements['n'],
                elements['e'],
                elements['peri0'],
                elements['dperi_dt'],
                elements['i'],
                elements['node0'],
                elements['dnode_dt'],
            ]
            if 'amp' in elements:   # only one wobble term is supported here
                items += [
                    elements['amp'],
                    elements['phase0'],
                    elements['dphase_dt']
                ]
            self.set_params(items)
        else:
            self.set_params(elements)

        if id is None:
            self.path_id = Path.temporary_path_id()
        else:
            self.path_id = id

        self.shape = ()
        self.keys = set()
        self.register()
    #===========================================================================



    #===========================================================================
    # set_params_new
    #===========================================================================
    def set_params_new(self, elements):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Re-define the path given new orbital elements.

        Part of the Fittable interface.

        Input:
            elements    An array or list of orbital elements. In order, they are
                            [a, mean0, dmean, e, peri0, dperi, i, node0, dnode],
                        followed by
                            [amp, phase0, dphase]
                        for each wobble.

              a         semimajor axis (km).
              mean0     mean longitude (radians) at the epoch.
              dmean     mean motion (radians/s).
              e         eccentricity.
              peri0     longitude of pericenter (radians) at the epoch.
              dperi     pericenter precession rate (rad/s).
              i         inclination (radians).
              node0     ascending node (radians) at the epoch.
              dnode     nodal regression rate (rad/s, < 0).

              amp       amplitude of the wobble, radians.
              phase0    phase of the wobble at epoch, radians.
              dphase    rate of change of the wobble, radians/s.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global NELEMENTS

        self.elements = np.asfarray(elements)
        assert self.elements.shape == (self.nparams,)

        #--------------------------------------------------------
        # Make copies of the orbital elements for convenience
        #--------------------------------------------------------
        self.a = self.elements[SEMIM]
        self.e = self.elements[ECCEN]
        self.i = self.elements[INCLI]
        self.cos_i = np.cos(self.i)
        self.sin_i = np.sin(self.i)

        self.mean0 = self.elements[MEAN0]
        self.peri0 = self.elements[PERI0]
        self.node0 = self.elements[NODE0]

        self.dmean_dt = self.elements[DMEAN]
        self.dperi_dt = self.elements[DPERI]
        self.dnode_dt = self.elements[DNODE]

        self.ae = self.a * self.e

        self.amp       = np.array(self.elements[NELEMENTS+LIBAMP::NWOBBLES])
        self.phase0    = np.array(self.elements[NELEMENTS+PHASE0::NWOBBLES])
        self.dphase_dt = np.array(self.elements[NELEMENTS+DPHASE::NWOBBLES])

        if self.amp.size == 0:      # because zero-sized arrays cause problems
            self.amp       = 0.
            self.phase0    = 0.
            self.dphase_dt = 0.

        #----------------------
        # Empty the cache
        #----------------------
        self.cached_observation_time = None
        self.cached_planet_event = None
    #===========================================================================



    #===========================================================================
    # copy
    #===========================================================================
    def copy(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return a deep copy of the object. Part of the Fittable interface.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Kepler(self.planet, self.epoch, self.get_params().copy(),
                      self.observer, self.wobbles)
    #===========================================================================



    #===========================================================================
    # get_elements
    #===========================================================================
    def get_elements(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the complete set of orbital elements, including wobbles.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.elements
    #===========================================================================



    #===========================================================================
    # xyz_planet
    #===========================================================================
    def xyz_planet(self, time, partials=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Body position and velocity relative to the planet, in planet's frame.

        Results are returned in an inertial frame where the Z-axis is aligned
        with the planet's rotation pole. Optionally, it also returns the partial
        derivatives of the position vector with respect to the orbital elements,
        on the the assumption that all orbital elements are independent. The
        The coordinates are only accurate to first order in (e,i) and in the
        wobbles. The derivatives are precise relative to the definitions of
        these elements. However, partials are not provided for the wobbles.

        Input:
            time        time (seconds) as a Scalar.
            partials    True to include partial derivatives of the position with
                        respect to the elements.

        Return:         (pos, vel)
            pos         a Vector3 of position vectors.
            vel         a Vector3 of velocity vectors.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        global SEMIM, MEAN0, DMEAN, ECCEN, PERI0, DPERI, INCLI, NODE0, DNODE
        global NELEMENTS

        global LIBAMP, PHASE0, DPHASE
        global NWOBBLES

        #------------------------------------
        # Convert to array if necessary
        #------------------------------------
        time = Scalar.as_scalar(time)
        t = time.vals - self.epoch

        if partials:
            partials_shape = time.shape + (self.nparams,)
            dmean_delem = np.zeros(partials_shape)
            dperi_delem = np.zeros(partials_shape)
            dnode_delem = np.zeros(partials_shape)
            da_delem = np.zeros(partials_shape)
            de_delem = np.zeros(partials_shape)
            di_delem = np.zeros(partials_shape)

        #------------------------------------------------------
        # Determine three angles and their time derivatives
        #   mean = mean0 + t * dmean_dt
        #   peri = peri0 + t * dperi_dt
        #   node = node0 + t * dnode_dt
        #------------------------------------------------------
        mean = self.mean0 + t * self.dmean_dt
        peri = self.peri0 + t * self.dperi_dt
        node = self.node0 + t * self.dnode_dt
        a = self.a
        e = self.e
        i = self.i

        #-------------------------
        # Time derivatives...
        #-------------------------
        dmean_dt = self.dmean_dt
        dperi_dt = self.dperi_dt
        dnode_dt = self.dnode_dt
        da_dt = 0.
        de_dt = 0.
        di_dt = 0.

        #----------------
        # Partials...
        #----------------
        if partials:
            dmean_delem[..., MEAN0] = 1.
            dmean_delem[..., DMEAN] = t
            dperi_delem[..., PERI0] = 1.
            dperi_delem[..., DPERI] = t
            dnode_delem[..., NODE0] = 1.
            dnode_delem[..., DNODE] = t
            da_delem[..., SEMIM] = 1.
            de_delem[..., ECCEN] = 1.
            di_delem[..., INCLI] = 1.

        #------------------------
        # Apply the wobbles
        #------------------------

        # For Laplace planes
        laplace_plane = False

        # For each wobble...
        start = NELEMENTS - NWOBBLES
        for k in range(self.nwobbles):
            start += NWOBBLES

            # 2-D librations
            if self.wobbles[k] in ('e2d', 'i2d'):
                if self.wobbles[k] == 'e2d':
                    amp = e
                    damp_dt = de_dt
                    angle = peri
                    dangle_dt = dperi_dt

                    if partials:
                        damp_delem = de_delem
                        dangle_delem = dperi_delem

                else:
                    amp = i
                    damp_dt = di_dt
                    angle = node
                    dangle_dt = dnode_dt

                    if partials:
                        damp_delem = di_delem
                        dangle_delem = dnode_delem

                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                x = amp * cos_angle
                y = amp * sin_angle
                dx_dt = damp_dt * cos_angle - y * dangle_dt
                dy_dt = damp_dt * sin_angle + x * dangle_dt

                if partials:
                    dx_delem = (damp_delem * cos_angle[...,np.newaxis] -
                                y[...,np.newaxis] * dangle_delem)
                    dy_delem = (damp_delem * sin_angle[...,np.newaxis] +
                                x[...,np.newaxis] * dangle_delem)

                arg = self.phase0[k] + t * self.dphase_dt[k]
                sin_arg = np.sin(arg)
                cos_arg = np.cos(arg)

                if partials:
                    darg_delem = np.zeros(partials_shape)
                    darg_delem[...,start+1] = 1.
                    darg_delem[...,start+2] = t

                x1 = self.amp[k] * cos_arg
                y1 = self.amp[k] * sin_arg
                dx1_dt = -y1 * self.dphase_dt[k]
                dy1_dt =  x1 * self.dphase_dt[k]

                if partials:
                    dx1_delem = np.zeros(partials_shape)
                    dy1_delem = np.zeros(partials_shape)
                    dx1_delem[...,start] = cos_arg
                    dy1_delem[...,start] = sin_arg

                    dx1_darg = -self.amp[k] * sin_arg
                    dy1_darg =  self.amp[k] * cos_arg
                    dx1_delem += dx1_darg[...,np.newaxis] * darg_delem
                    dy1_delem += dy1_darg[...,np.newaxis] * darg_delem

                x2 = x + x1
                y2 = y + y1
                dx2_dt = dx_dt + dx1_dt
                dy2_dt = dy_dt + dy1_dt

                if partials:
                    dx2_delem = dx_delem + dx1_delem
                    dy2_delem = dy_delem + dy1_delem

                amp2 = np.sqrt(x2**2 + y2**2)
                damp2_dx2 = -x2 / amp2
                damp2_dy2 = -y2 / amp2
                damp2_dt = damp2_dx2 * dx2_dt + damp2_dy2 * dy2_dt

                angle2 = np.arctan2(y2,x2)
                dangle2_dx2 =  x2 / (x2**2 + y2**2)
                dangle2_dy2 = -y2 / (x2**2 + y2**2)
                dangle2_dt = dangle2_dx2 * dx2_dt + dangle2_dy2 * dy2_dt

                if partials:
                    damp2_delem = (damp2_dx2[...,np.newaxis] * dx2_delem +
                                   damp2_dy2[...,np.newaxis] * dy2_delem)

                    dangle2_delem = (dangle2_dx2[...,np.newaxis] * dx2_delem +
                                     dangle2_dy2[...,np.newaxis] * dy2_delem)

                if self.wobbles[k] == 'e2d':
                    e = amp2
                    de_dt = damp2_dt
                    peri = angle2
                    dperi_dt = dangle2_dt

                    if partials:
                        de_delem = damp2_delem
                        dperi_delem = dangle2_delem

                else:
                    i = amp2
                    di_dt = damp2_dt
                    node = angle2
                    dnode_dt = dangle2_dt

                    if partials:
                        di_delem = damp2_delem
                        dnode_delem = dangle2_delem

            # Single-element librations
            elif self.wobbles[k] in ('mean', 'peri', 'node', 'a', 'e', 'i'):

                arg = self.phase0[k] + t * self.dphase_dt[k]
                sin_arg = np.sin(arg)
                cos_arg = np.cos(arg)

                w = self.amp[k] * cos_arg
                dw_dt = -self.amp[k] * sin_arg * self.dphase_dt[k]

                if partials:
                    dw_delem = np.zeros(partials_shape)
                    dw_delem[...,start] = cos_arg
                    dw_delem[...,start+1] = self.amp[k] * cos_arg[k]
                    dw_delem[...,start+2] = dw_delem[...,start+1] * t

                if self.wobbles[k] == 'mean':
                    mean     += w
                    dmean_dt += dw_dt
                    if partials:
                        dmean_delem += dw_delem

                elif self.wobbles[k] == 'peri':
                    peri     += w
                    dperi_dt += dw_dt
                    if partials:
                        dperi_delem += dw_delem

                elif self.wobbles[k] == 'node':
                    node     += w
                    dnode_dt += dw_dt
                    if partials:
                        dnode_delem += dw_delem

                elif self.wobbles[k] == 'a':
                    a     += w
                    da_dt += dw_dt
                    if partials:
                        da_delem += dw_delem

                elif self.wobbles[k] == 'e':
                    e     += w
                    de_dt += dw_dt
                    if partials:
                        de_delem += dw_delem

                else:
                    pole_inc = w
                    di_dt += dw_dt
                    if partials:
                        di_delem += dw_delem

            # Laplace plane case
            else:
                arg = self.phase0[k] + t * self.dphase_dt[k]
                sin_arg = np.sin(arg)
                cos_arg = np.cos(arg)

                laplace_plane = True
                laplace_sin_inc = np.sin(self.amp[k])
                laplace_cos_inc = np.cos(self.amp[k])
                laplace_sin_node = sin_arg
                laplace_cos_node = cos_arg

        #----------------------------------
        # Evaluate some derived elements
        #----------------------------------
        ae = a * e
        cos_i = np.cos(i)
        sin_i = np.sin(i)

        # Time-derivatives...
        dae_dt = a * de_dt + da_dt * e
        dcosi_dt = -sin_i * di_dt
        dsini_dt =  cos_i * di_dt

        # Partials...
        if partials:
            dae_delem = np.zeros(partials_shape)
            dae_delem[..., SEMIM] = e
            dae_delem[..., ECCEN] = a

            dcosi_delem = np.zeros(partials_shape)
            dsini_delem = np.zeros(partials_shape)
            dcosi_delem[...,INCLI] = -sin_i
            dsini_delem[...,INCLI] =  cos_i

        #--------------------------------------------------
        # Determine moon polar coordinates in orbit plane
        #   r     = a - a * e * cos(mean - peri))
        #   theta = mean -  2 * e * sin(mean - peri)
        #--------------------------------------------------
        mp = mean - peri
        cos_mp = np.cos(mp)
        sin_mp = np.sin(mp)

        r     = a - ae * cos_mp
        theta = mean + 2. * e * sin_mp

        # Time-derivatives...
        dmp_dt = dmean_dt - dperi_dt
        dcosmp_dt = -sin_mp * dmp_dt
        dsinmp_dt =  cos_mp * dmp_dt

        dr_dt = da_dt - ae * dcosmp_dt - dae_dt * cos_mp
        dtheta_dt = dmean_dt + 2. * (e * dsinmp_dt + de_dt * sin_mp)

        # Partials...
        if partials:
            dmp_delem = dmean_delem - dperi_delem
            dcosmp_delem = -sin_mp[...,np.newaxis] * dmp_delem
            dsinmp_delem =  cos_mp[...,np.newaxis] * dmp_delem

            dr_delem = da_delem - ae[...,np.newaxis] * dcosmp_delem - \
                                  dae_delem * cos_mp[...,np.newaxis]
            dtheta_delem = dmean_delem + 2. * \
                                        (e[...,np.newaxis] * dsinmp_delem +
                                         de_delem * sin_mp[...,np.newaxis])

        #--------------------------------------------------------------------
        # Locate body on an inclined orbit, in a frame where X is along the
        # ascending node
        #   asc[X] = r cos(theta - node)
        #   asc[Y] = r sin(theta - node) cos(i)
        #   asc[Z] = r sin(theta - node) sin(i)
        #--------------------------------------------------------------------
        tn = theta - node
        cos_tn = np.cos(tn)
        sin_tn = np.sin(tn)

        unit1 = np.empty((time.shape + (3,)))
        unit1[...,0] = cos_tn
        unit1[...,1] = sin_tn * cos_i
        unit1[...,2] = sin_tn * sin_i

        asc = r[...,np.newaxis] * unit1

        # Time-derivatives...
        dtn_dt = dtheta_dt - dnode_dt
        dcostn_dt = -sin_tn * dtn_dt
        dsintn_dt =  cos_tn * dtn_dt

        dunit1_dt = np.empty(time.shape + (3,))
        dunit1_dt[...,0] = dcostn_dt
        dunit1_dt[...,1] = dsintn_dt * cos_i + sin_tn * dcosi_dt
        dunit1_dt[...,2] = dsintn_dt * sin_i + sin_tn * dsini_dt

        dasc_dt = dr_dt[...,np.newaxis] * unit1 + r[...,np.newaxis] * dunit1_dt

        # Partials...
        if partials:
            dtn_delem = dtheta_delem - dnode_delem
            dcostn_delem = -sin_tn[...,np.newaxis] * dtn_delem
            dsintn_delem =  cos_tn[...,np.newaxis] * dtn_delem

            dunit1_delem = np.empty(partials_shape + (3,))
            dunit1_delem[...,0] = dcostn_delem
            dunit1_delem[...,1] = dsintn_delem * cos_i[...,np.newaxis] + \
                                  sin_tn[...,np.newaxis] * dcosi_delem
            dunit1_delem[...,2] = dsintn_delem * sin_i[...,np.newaxis] + \
                                  sin_tn[...,np.newaxis] * dsini_delem

            dasc_delem = dr_delem[...,np.newaxis] * unit1[...,np.newaxis,:] + \
                         r[...,np.newaxis,np.newaxis] * dunit1_delem
            # shape is (..., 9, 3)

        #---------------------------------------------------------------------
        # Rotate the ascending node back into position in our inertial frame
        #   xyz[X] = asc[X] * cos(node) - asc[Y] * sin(node)
        #   xyz[Y] = asc[X] * sin(node) + asc[Y] * cos(node)
        #   xyz[Z] = asc[Z]
        #---------------------------------------------------------------------
        cos_node = np.cos(node)
        sin_node = np.sin(node)

        rotate = np.zeros(time.shape + (3,3))
        rotate[...,0,0] =  cos_node
        rotate[...,0,1] = -sin_node
        rotate[...,1,0] =  sin_node
        rotate[...,1,1] =  cos_node
        rotate[...,2,2] =  1.

        xyz = np.sum(rotate[...,:,:] * asc[...,np.newaxis,:], axis=-1)

        # Time-derivatives...
        dcosnode_dt = -sin_node * dnode_dt
        dsinnode_dt =  cos_node * dnode_dt

        drotate_dt = np.zeros(time.shape + (3,3))
        drotate_dt[...,0,0] =  dcosnode_dt
        drotate_dt[...,0,1] = -dsinnode_dt
        drotate_dt[...,1,0] =  dsinnode_dt
        drotate_dt[...,1,1] =  dcosnode_dt

        dxyz_dt = (np.sum(rotate * dasc_dt[...,np.newaxis,:], axis=-1) +
                   np.sum(drotate_dt * asc[...,np.newaxis,:], axis=-1))

        # Partials...
        if partials:
            dcosnode_delem = -sin_node[...,np.newaxis] * dnode_delem
            dsinnode_delem =  cos_node[...,np.newaxis] * dnode_delem

            drotate_delem = np.zeros(partials_shape + (3,3))
            drotate_delem[...,0,0] =  dcosnode_delem
            drotate_delem[...,0,1] = -dsinnode_delem
            drotate_delem[...,1,0] =  dsinnode_delem
            drotate_delem[...,1,1] =  dcosnode_delem

            dxyz_delem = np.sum(rotate[...,np.newaxis,:,:] *
                                dasc_delem[...,np.newaxis,:], axis=-1) + \
                         np.sum(drotate_delem *
                                asc[...,np.newaxis,np.newaxis,:], axis=-1)
            # shape = (..., 9, 3)

        #------------------------------------------
        # Apply Laplace Plane
        #   asc[X] = r cos(theta - node)
        #   asc[Y] = r sin(theta - node) cos(i)
        #   asc[Z] = r sin(theta - node) sin(i)
        #------------------------------------------
        if laplace_plane:
            node = np.array([laplace_cos_node, laplace_sin_node, 0.])

            cos_sin_node = laplace_cos_node * laplace_sin_node
            cos2_node = laplace_cos_node**2
            sin2_node = laplace_sin_node**2

            cos_sin_node_1_minus_cos_inc = (laplace_cos_node *
                                            laplace_sin_node *
                                            (1. - laplace_cos_inc))

            rotate = np.array(
                [[cos2_node + sin2_node * laplace_cos_inc,
                  cos_sin_node_1_minus_cos_inc,
                  laplace_sin_node * laplace_cos_inc],
                 [cos_sin_node_1_minus_cos_inc,
                  sin2_node + cos2_node * laplace_cos_inc,
                  -laplace_cos_node * laplace_sin_inc],
                 [-laplace_sin_node * laplace_sin_inc,
                   laplace_cos_node * laplace_sin_inc,
                   laplace_cos_inc]
                ])

            xyz = np.sum(rotate[...,:,:] * xyz[...,np.newaxis,:], axis=-1)
            dxyz_dt = np.sum(rotate[...,:,:] * dxyz_dt[...,np.newaxis,:],
                             axis=-1)

            if partials:
                dxyz_delem = np.sum(rotate[...,np.newaxis,:,:] *
                                    dxyz_delem[...,np.newaxis,:], axis=-1)

        #-------------------
        # Return results
        #-------------------
        pos = Vector3(xyz)
        vel = Vector3(dxyz_dt)

        if partials:
            dxyz_delem = dxyz_delem.swapaxes(-1,-2)
            pos.insert_deriv('elements', Vector3(dxyz_delem, drank=1))

        return (pos, vel)
    #===========================================================================



    #===========================================================================
    # xyz_observed
    #===========================================================================
    def xyz_observed(self, time, quick={}, partials=False, planet_event=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Body position and velocity relative to the observer in J2000 frame.

        Input:
            time        time (seconds) as a Scalar.
            partials    True to include partial derivatives of the position with
            quick       False to disable QuickPaths; a dictionary to override
                        specific options.
                        respect to the elements.
            planet_event
                        the corresponding event of the photon leaving the
                        planet; None to calculate this from the time. Note that
                        this can be calculated once using planet_event(),
                        avoiding the need to re-calculate this quantity for
                        repeated calls using the same time(s).

        Return:
            pos         a Vector3 of position vectors.
            vel         a Vector3 of velocity vectors.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if planet_event is None:
            observer_event = Event(time, Vector3.ZERO,
                                   self.observer, self.frame)
            planet_event = self.center.photon_to_event(observer_event,
                                                       quick=quick)[0]

        (pos, vel) = self.xyz_planet(planet_event.time, partials)

        pos_j2000 = self.to_j2000.rotate(pos) + planet_event.pos
        vel_j2000 = self.to_j2000.rotate(vel) + planet_event.vel

        return (pos_j2000, vel_j2000)
    #===========================================================================



    #===========================================================================
    # event_at_time
    #===========================================================================
    def event_at_time(self, time, quick={}, partials=False, planet_event=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return an Event object corresponding to a Scalar time on this path.

        Input:
            time        a time Scalar at which to evaluate the path.
            quick       False to disable QuickPaths; a dictionary to override
                        specific options.
            partials    True to include the derivatives of position with respect
                        to the orbital elements; False otherwise.
            planet_event  optional event of the photon leaving the center of the
                        planet. Saves a re-calculation if the time is re-used.
                        Only relevant when an observer is defined.

        Return:         an Event object containing the time, position and
                        velocity of the paths.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #----------------------------------------------------------
        # Without an observer, return event in the planet frame
        #----------------------------------------------------------
        if self.observer is None:
            (pos, vel) = self.xyz_planet(time, partials=partials)
            return Event(time, (pos, vel), self.origin, self.frame)

        #--------------------------------------------------
        # Otherwise, return the event WRT the observer
        #--------------------------------------------------
        (pos, vel) = self.xyz_observed(time, quick, partials, planet_event)
        return Event(time, (pos, vel), self.observer, Frame.J2000)
    #===========================================================================



    #===========================================================================
    # node_at_time
    #===========================================================================
    def node_at_time(self, time):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the longitude of ascending node at the specified time. Wobbles
        are ignored. The angle is a positive rotation about the planet's ring
        frame.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        global NODE0, DNODE

        time = Scalar.as_scalar(time)
        return self.elements[NODE0] + (time-self.epoch) * self.elements[DNODE]
    #===========================================================================



    #===========================================================================
    # pole_at_time
    #===========================================================================
    def pole_at_time(self, time):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the J2000 vector pointing toward the orbit's pole at the
        specified time. Wobbles are ignored.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        xform = self.frame_wrt_j2000.transform_at_time(time)
        x_axis_in_j2000 = xform.unrotate(Vector3.XAXIS)
        y_axis_in_j2000 = xform.unrotate(Vector3.YAXIS)
        z_axis_in_j2000 = xform.unrotate(Vector3.ZAXIS)

        node = self.node_at_time(time)
        cos_node = np.cos(node)
        sin_node = np.sin(node)
        node_in_j2000 = (cos_node * x_axis_in_j2000 +
                         sin_node * y_axis_in_j2000)

        #-----------------------------------------------------------------------
        # This vector is 90 degrees behind of the node in the reference equator
        #-----------------------------------------------------------------------
        target_in_j2000 = ( sin_node * x_axis_in_j2000 +
                           -cos_node * y_axis_in_j2000)

        return self.cos_i * z_axis_in_j2000 + self.sin_i * target_in_j2000
    #===========================================================================



    ####################################################
    # Override for the case where observer != None
    ####################################################

    #===========================================================================
    # photon_to_event
    #===========================================================================
    def photon_to_event(self, arrival, derivs=False, guess=None, quick={},
                              converge={}, partials=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The photon departure event from this path to match the arrival event.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.observer is None:
            (path_event,
             link_event) = super(Kepler,
                                 self).photon_to_event(arrival, derivs, guess,
                                                       quick=quick,
                                                       converge=converge)
            if partials:
                (pos, vel) = self.xyz_planet(event.time, partials=True)
                path_event.pos.insert_deriv('elements', pos.d_delements)

            return (path_event, link_event)

        (planet_event,
         link_event) = self.center.photon_to_event(arrival, derivs, guess,
                                                   quick, converge)

        path_event = self.event_at_time(planet_event.time, quick, partials,
                                        planet_event)

        path_event.dep_lt = planet_event.time - link_event.time
        path_event.dep_j2000 = path_event.pos_j2000 - link_event.pos_j2000

        link_event = Event(link_event.time, link_event.state,
                           link_event.origin, link_event.frame)
        link_event.arr_lt = path_event.dep_lt
        link_event.arr_j2000 = path_event.dep_j2000

        return (path_event, link_event)
    #===========================================================================



#*******************************************************************************



################################################################################
# UNIT TESTS
################################################################################

import unittest

#===============================================================================
# _xyz_planet_derivative_test
#===============================================================================
def _xyz_planet_derivative_test(kep, t, delta=1.e-7):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Error in position change based on numerical vs. analytic derivatives.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #------------------------------------------
    # Save the position and its derivatives
    #------------------------------------------
    (xyz, d_xyz_dt) = kep.xyz_planet(t, partials=True)
    d_xyz_d_elem = xyz.d_delements.vals
    pos_norm = xyz.norm().vals
    vel_norm = d_xyz_dt.norm().vals

    #----------------------------------------------------------
    # Create new Kepler objects for tweaking the parameters
    #----------------------------------------------------------
    khi = kep.copy()
    klo = kep.copy()

    params = kep.get_params()

    #---------------------------------
    # Loop through parameters...
    #---------------------------------
    errors = np.zeros(np.shape(t) + (3,kep.nparams))
    for e in range(kep.nparams):

        # Tweak one parameter
        hi = params.copy()
        lo = params.copy()

        if params[e] == 0.:
            hi[e] += delta
            lo[e] -= delta
        else:
            hi[e] *= 1. + delta
            lo[e] *= 1. - delta

        denom = hi[e] - lo[e]

        khi.set_params(hi)
        klo.set_params(lo)

        # Compare the change with that derived from the partial derivative
        xyz_hi = khi.xyz_planet(t, partials=False)[0].vals
        xyz_lo = klo.xyz_planet(t, partials=False)[0].vals
        hi_lo_diff = xyz_hi - xyz_lo

        errors[...,:,e] = ((d_xyz_d_elem[...,:,e] * denom - hi_lo_diff) /
                           pos_norm[...,np.newaxis])

    return errors
#===============================================================================



#===============================================================================
# _pos_derivative_test
#===============================================================================
def _pos_derivative_test(kep, t, delta=1.e-5):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Calculates numerical derivatives of (x,y,z) in the observer/J2000 frame
    relative to the orbital elements, at time(s) t. It returns a tuple of
    (numerical derivatives, analytic derivatives, relative errors). Used for
    debugging.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #-------------------------------------------
    # Save the position and its derivatives
    #-------------------------------------------
    event = kep.event_at_time(t, partials=True)
    xyz = event.pos.vals
    d_xyz_dt = event.vel.vals
    d_xyz_d_elem = event.pos.d_delements.vals

    pos_norm = event.pos.norm().vals
    vel_norm = event.vel.norm().vals

    #----------------------------------------------------------
    # Create new Kepler objects for tweaking the parameters
    #----------------------------------------------------------
    khi = kep.copy()
    klo = kep.copy()

    params = kep.get_params()

    #-------------------------------
    # Loop through parameters...
    #-------------------------------
    new_derivs = np.zeros(np.shape(t) + (3,kep.nparams))
    errors = np.zeros(np.shape(t) + (3,kep.nparams))
    for e in range(kep.nparams):

        # Tweak one parameter
        hi = params.copy()
        lo = params.copy()

        if params[e] == 0.:
            hi[e] += delta
            lo[e] -= delta
        else:
            hi[e] *= 1. + delta
            lo[e] *= 1. - delta

        denom = hi[e] - lo[e]

        khi.set_params(hi)
        klo.set_params(lo)

        # Compare the change with that derived from the partial derivative
        xyz_hi = khi.event_at_time(t, partials=False).pos.vals
        xyz_lo = klo.event_at_time(t, partials=False).pos.vals
        hi_lo_diff = xyz_hi - xyz_lo

        errors[...,:,e] = ((d_xyz_d_elem[...,:,e] * denom - hi_lo_diff) /
                           pos_norm[...,np.newaxis])

    return errors
#===============================================================================



#*******************************************************************************
# Test_Kepler
#*******************************************************************************
class Test_Kepler(unittest.TestCase):

    #===========================================================================
    # setUp
    #===========================================================================
    def setUp(self):
        import oops.body as body

        body.Body.reset_registry()
        body.define_solar_system("2000-01-01", "2010-01-01")
    #===========================================================================



    #===========================================================================
    # tearDown
    #===========================================================================
    def tearDown(self):
        pass
    #===========================================================================



    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):
        import oops.body as body

        # SEMIM = 0    elements[SEMIM] = semimajor axis (km)
        # MEAN0 = 1    elements[MEAN0] = mean longitude at epoch (radians)
        # DMEAN = 2    elements[DMEAN] = mean motion (radians/s)
        # ECCEN = 3    elements[ECCEN] = eccentricity
        # PERI0 = 4    elements[PERI0] = pericenter at epoch (radians)
        # DPERI = 5    elements[DPERI] = pericenter precession rate (radians/s)
        # INCLI = 6    elements[INCLI] = inclination (radians)
        # NODE0 = 7    elements[NODE0] = longitude of ascending node at epoch
        # DNODE = 8    elements[DNODE] = nodal regression rate (radians/s)

        a = 140000.

        saturn = Gravity.lookup('SATURN')
        dmean_dt = saturn.n(a)
        dperi_dt = saturn.dperi_dt(a)
        dnode_dt = saturn.dnode_dt(a)

        TIMESTEPS = 100
        time = 3600. * np.arange(TIMESTEPS)

        kep = Kepler(body.Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt, 0.2, 3., dperi_dt, 0.1, 5., dnode_dt),
                       Path.as_path("EARTH"))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-8)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-8)

        ####################

        kep = Kepler(body.Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        dmean_dt * 0.10, 2., dmean_dt / 100.,
                        dperi_dt * 0.08, 4., dmean_dt / 50.,
                        dnode_dt * 0.12, 6., dmean_dt / 200.),
                       Path.as_path("EARTH"), wobbles=('mean', 'peri', 'node'))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-8)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-8)

        ####################

        kep = Kepler(body.Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        a * 0.10, 2., dmean_dt / 100.),
                       Path.as_path("EARTH"), wobbles=('a',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-7)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        ####################

        kep = Kepler(body.Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        0.1, 4., dmean_dt / 50.),
                       Path.as_path("EARTH"), wobbles=('e',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 3.e-7)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 3.e-5)

        ####################

        kep = Kepler(body.Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        0.15, 2., dmean_dt / 150.),
                       Path.as_path("EARTH"), wobbles=('i',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-7)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        ####################

        kep = Kepler(body.Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        1.e-4, 3., dperi_dt/100.),
                       Path.as_path("EARTH"), wobbles=('e2d',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-4)

        ####################

        kep = Kepler(body.Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        1.e-4, 2., dnode_dt/150.),
                       Path.as_path("EARTH"), wobbles=('i2d',))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-6)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        ####################

        kep = Kepler(body.Body.lookup("SATURN"), 0.,
                       (a, 1., dmean_dt,
                        0.2, 3., dperi_dt,
                        0.1, 5., dnode_dt,
                        1.e-4, 2., dperi_dt/150.,
                        2.e-4, 3., dnode_dt/200.,
                        a * 1.e-3, 4., dmean_dt/150.),
                       Path.as_path("EARTH"), wobbles=('i2d','e2d','a'))

        errors = _xyz_planet_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-5)

        errors = _pos_derivative_test(kep, time)
        self.assertTrue(np.max(np.abs(errors)) < 1.e-4)

        Frame.reset_registry()
        Path.reset_registry()
        body.Body.reset_registry()
    #===========================================================================


#*******************************************************************************


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
