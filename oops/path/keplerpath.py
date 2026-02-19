##########################################################################################
# oops/path/keplerpath.py: Subclass KeplerPath of class Path.
##########################################################################################

import numpy as np

from polymath          import Scalar, Vector3, Matrix3
from oops.cache        import Cache
from oops.event        import Event
from oops.fittable     import Fittable
from oops.frame.frame_ import Frame
from oops.path.path_   import Path

_SEMIM = 0      # elements[SEMIM] = semimajor axis (km)
_MEAN0 = 1      # elements[MEAN0] = mean longitude at epoch (radians)
_DMEAN = 2      # elements[DMEAN] = mean motion (radians/s)
_ECCEN = 3      # elements[ECCEN] = eccentricity
_PERI0 = 4      # elements[PERI0] = pericenter at epoch (radians)
_DPERI = 5      # elements[DPERI] = pericenter precession rate (radians/s)
_INCLI = 6      # elements[INCLI] = inclination (radians)
_NODE0 = 7      # elements[NODE0] = longitude of ascending node at epoch
_DNODE = 8      # elements[DNODE] = nodal regression rate (radians/s, negative)

_NELEMENTS = 9

_LIBAMP = 0     # libration[LIBAMP] = libration amplitude (radians)
_PHASE0 = 1     # libration[PHASE0] = libration phase at epoch (radians)
_DPHASE = 2     # libration[DPHASE] = libration rate (radians/s)

_NWOBBLES = 3


class KeplerPath(Path, Fittable):
    """A Path subclass that defines a fittable Keplerian orbit.

    It is accurate to first order in eccentricity and inclination, and is defined using
    nine orbital elements.
    """

    _WAYPOINTS = {}

    def __init__(self, body, epoch, elements=None, observer=None, *, wobbles=(),
                 path_id=None):
        """Constructor for a KeplerPath.

        Parameters:
            body (Body or str): The Body object or name of the central planet, including
                its gravity and its ring_frame.
            epoch (float): The time TDB relative to which all orbital elements are
                 defined.
            elements (array-like or dict, optional): The orbital elements and wobble
                terms. If an array-like object is provided, this is the order of the
                elements:

                * [0]: `a`, mean radius of orbit, km.
                * [1]: `lon`, mean longitude of orbit at epoch, radians.
                * [2]: `n`, mean motion, radians/sec.
                * [3]: `e`, orbital eccentricity.
                * [4]: `peri`, longitude of pericenter at epoch, radians.
                * [5]: `prec`,pericenter precession rate, radians/sec.
                * [6]: `i`, inclination, radians.
                * [7]: `node`, longitude of ascending node at epoch, radians.
                * [8]: `regr`, nodal regression rate, radians/sec, NEGATIVE!

                You can include additional "wobble" terms, which can describe
                non-Keplerian orbital perturbations. Repeat these three elements for each
                wobble term:

                * [9, 12, ...] `amp`: amplitude of the term, radians.
                * [10, 13, ...] `phase0`: initial phase of the first wobble term, radians.
                * [11, 14, ...] `dphase_dt`: rate of change of the first wobble term,
                  radians/s.

                Alternatively, provide a dictionary containing keys with these names (in
                which case only one wobble term is allowed). If the elements are not
                provided, the object remains un-initialized until `set_elements` is
                called.
            observer (Path or str, optional): Identification of the Path of the observer.
                If provided, then `event_at_time` returns positions relative to this
                observer in J2000 coordinates and with light travel time already accounted
                for; this makes it easy to use this Path object for astrometry and orbit
                fitting. If not provided, `event_at_time` returns positions relative to
                the planet center and in the planet's `ring_frame`.
            wobbles (str or tuple): The name(s) of each wobble element:

                * "a": semimajor axis.
                * "e": eccentricity.
                * "i": inclination.
                * "mean": mean motion rate.
                * "peri": pericenter.
                * "node": ascending node.
                * "e2d": forced eccentricity represented by a 2-D vector.
                * "i2d": forced inclination represented by a 2-D vector.
                * "pole": an offset to the Laplace plane.

            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.
        """

        self._wobbles = (wobbles,) if isinstance(wobbles, str) else wobbles
        self._nwobbles = len(self._wobbles)
        for name in self._wobbles:
            if name not in {'mean', 'peri', 'node', 'a', 'e', 'i', 'e2d', 'i2d', 'pole'}:
                raise ValueError(f'invalid name for wobble in KeplerPath: {name}')

        self._nelements = _NELEMENTS + self._nwobbles * _NWOBBLES
        self.nparams = self._nelements

        self._planet = Path._Body.as_body(body)
        self._center = self._planet.path
        self._gravity = self._planet.gravity
        self._epoch = float(epoch)
        self._events = Cache()

        if observer is None:
            self._observer = None
            self._origin = self.planet.path
            self._frame = self.planet.ring_frame
            self._to_j2000 = Matrix3.IDENTITY
        else:
            self._observer = Path.as_waypoint(observer)
            if self._observer._shape:
                raise ValueError('KeplerPath requires a shapeless observer')

            self._origin = self._observer
            self._frame = Frame.J2000
            frame = Frame.J2000.wrt(self._planet.ring_frame)
            self.to_j2000 = frame.transform_at_time(self._epoch).matrix

        if elements is None:
            self._elements = None
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
            self.set_elements(items)
        else:
            self.set_elements(elements)

        self._shape = ()

        self._register(path_id)
        self.refresh()

    def set_elements(self, elements):
        """Re-define the path given new orbital elements.

        Parameters:
            elements (array-like): The orbital elements and wobble terms, in this order:

                * [0]: `a`, mean radius of orbit, km.
                * [1]: `lon`, mean longitude of orbit at epoch, radians.
                * [2]: `n`, mean motion, radians/sec.
                * [3]: `e`, orbital eccentricity.
                * [4]: `peri`, longitude of pericenter at epoch, radians.
                * [5]: `prec`,pericenter precession rate, radians/sec.
                * [6]: `i`, inclination, radians.
                * [7]: `node`, longitude of ascending node at epoch, radians.
                * [8]: `regr`, nodal regression rate, radians/sec, NEGATIVE!

                If the orbit has "wobble" terms, which can describe streamlines
                of ring particles. Repeat these three elements for each wobble term:

                * [9, 12, ...] `amp`: amplitude of the term, radians.
                * [10, 13, ...] `phase0`: initial phase of the first wobble term, radians.
                * [11, 14, ...] `dphase_dt`: rate of change of the first wobble term,
                  radians/s.
        """

        self._elements = np.asarray(elements).copy()

        # Make copies of the orbital elements for convenience
        self._a = self._elements[_SEMIM]
        self._e = self._elements[_ECCEN]
        self._i = self._elements[_INCLI]
        self._cos_i = np.cos(self._i)
        self._sin_i = np.sin(self._i)

        self._mean0 = self._elements[_MEAN0]
        self._peri0 = self._elements[_PERI0]
        self._node0 = self._elements[_NODE0]

        self._dmean_dt = self._elements[_DMEAN]
        self._dperi_dt = self._elements[_DPERI]
        self._dnode_dt = self._elements[_DNODE]

        self._ae = self._a * self._e

        self._amp       = np.array(self._elements[_NELEMENTS+_LIBAMP::_NWOBBLES])
        self._phase0    = np.array(self._elements[_NELEMENTS+_PHASE0::_NWOBBLES])
        self._dphase_dt = np.array(self._elements[_NELEMENTS+_DPHASE::_NWOBBLES])

        if self._amp.size == 0:     # because zero-sized arrays cause problems
            self._amp       = 0.
            self._phase0    = 0.
            self._dphase_dt = 0.

    def get_elements(self):
        return self._elements

    def _waypoint_key(self):
        if self.is_frozen:
            return (self._planet, self._epoch, self._elements, self._observer,
                    self._wobbles)
        # Use id(self) to ensure that an un-frozen KeplerPath has a unique key
        return id(self)

    ######################################################################################
    # Fittable support
    ######################################################################################

    def _set_params(self, params):
        """Re-define the orbital elements of this KeplerPath."""
        self._elements = params

    @property
    def params(self):
        return tuple(self._elements)

    def _refresh(self):
        self.set_elements(self._elements)

    def _freeze(self):
        self._reregister()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._planet, self._epoch, self._elements, self._observer, self._wobbles,
                self.stripped_id)

    def __setstate__(self, state):
        (body, epoch, elements, observer, wobbles, path_id) = state
        self.__init__(body, epoch, elements, observer, wobbles=wobbles, path_id=path_id)
        self.freeze()

    ######################################################################################
    # Orbit calculation relative to planet
    ######################################################################################

    def _xyz_planet(self, time, partials=False):
        """Body position and velocity relative to the planet, in planet's frame.

        Results are returned in an inertial frame where the Z-axis is aligned with the
        planet's rotation pole. Optionally, it also returns the partial derivatives of the
        position vector with respect to the orbital elements, on the the assumption that
        all orbital elements are independent. The The coordinates are only accurate to
        first order in (e,i) and in the wobbles. The derivatives are precise relative to
        the definitions of these elements. However, partials are not provided for the
        wobbles.

        Parameters:
            time (Scalar or float): Time in seconds TDB.
            partials (bool, optional): True to include partial derivatives of the position
                with respect to the elements.

        Returns:
            (tuple): (position, velocity), each represented by a Vector3.
        """

        # Convert to array if necessary
        time = Scalar.as_scalar(time)
        t = time.vals - self._epoch

        if partials:
            partials_shape = time.shape + (self._nelements,)
            dmean_delem = np.zeros(partials_shape)
            dperi_delem = np.zeros(partials_shape)
            dnode_delem = np.zeros(partials_shape)
            da_delem = np.zeros(partials_shape)
            de_delem = np.zeros(partials_shape)
            di_delem = np.zeros(partials_shape)

        ##################################################################################
        # Determine three angles and their time derivatives
        #   mean = mean0 + t * dmean_dt
        #   peri = peri0 + t * dperi_dt
        #   node = node0 + t * dnode_dt
        ##################################################################################

        mean = self._mean0 + t * self._dmean_dt
        peri = self._peri0 + t * self._dperi_dt
        node = self._node0 + t * self._dnode_dt
        a = self._a
        e = self._e
        i = self._i

        # Time derivatives...
        dmean_dt = self._dmean_dt
        dperi_dt = self._dperi_dt
        dnode_dt = self._dnode_dt
        da_dt = 0.
        de_dt = 0.
        di_dt = 0.

        # Partials...
        if partials:
            dmean_delem[..., _MEAN0] = 1.
            dmean_delem[..., _DMEAN] = t
            dperi_delem[..., _PERI0] = 1.
            dperi_delem[..., _DPERI] = t
            dnode_delem[..., _NODE0] = 1.
            dnode_delem[..., _DNODE] = t
            da_delem[..., _SEMIM] = 1.
            de_delem[..., _ECCEN] = 1.
            di_delem[..., _INCLI] = 1.

        ##################################################################################
        # Apply the wobbles
        ##################################################################################

        # For Laplace planes
        laplace_plane = False

        # For each wobble...
        start = _NELEMENTS - _NWOBBLES
        for k in range(self._nwobbles):
            start += _NWOBBLES

            # 2-D librations
            if self._wobbles[k] in ('e2d', 'i2d'):
                if self._wobbles[k] == 'e2d':
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
                    dx_delem = (damp_delem * cos_angle[...,np.newaxis]
                                - y[..., np.newaxis] * dangle_delem)
                    dy_delem = (damp_delem * sin_angle[...,np.newaxis]
                                + x[..., np.newaxis] * dangle_delem)

                arg = self._phase0[k] + t * self._dphase_dt[k]
                sin_arg = np.sin(arg)
                cos_arg = np.cos(arg)

                if partials:
                    darg_delem = np.zeros(partials_shape)
                    darg_delem[...,start+1] = 1.
                    darg_delem[...,start+2] = t

                x1 = self._amp[k] * cos_arg
                y1 = self._amp[k] * sin_arg
                dx1_dt = -y1 * self.dphase_dt[k]
                dy1_dt =  x1 * self.dphase_dt[k]

                if partials:
                    dx1_delem = np.zeros(partials_shape)
                    dy1_delem = np.zeros(partials_shape)
                    dx1_delem[..., start] = cos_arg
                    dy1_delem[..., start] = sin_arg

                    dx1_darg = -self._amp[k] * sin_arg
                    dy1_darg =  self._amp[k] * cos_arg
                    dx1_delem += dx1_darg[..., np.newaxis] * darg_delem
                    dy1_delem += dy1_darg[..., np.newaxis] * darg_delem

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

                angle2 = np.arctan2(y2, x2)
                dangle2_dx2 =  x2 / (x2**2 + y2**2)
                dangle2_dy2 = -y2 / (x2**2 + y2**2)
                dangle2_dt = dangle2_dx2 * dx2_dt + dangle2_dy2 * dy2_dt

                if partials:
                    damp2_delem = (damp2_dx2[..., np.newaxis] * dx2_delem +
                                   damp2_dy2[..., np.newaxis] * dy2_delem)

                    dangle2_delem = (dangle2_dx2[..., np.newaxis] * dx2_delem +
                                     dangle2_dy2[..., np.newaxis] * dy2_delem)

                if self._wobbles[k] == 'e2d':
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
            elif self._wobbles[k] in ('mean', 'peri', 'node', 'a', 'e', 'i'):

                arg = self._phase0[k] + t * self._dphase_dt[k]
                sin_arg = np.sin(arg)
                cos_arg = np.cos(arg)

                w = self._amp[k] * cos_arg
                dw_dt = -self._amp[k] * sin_arg * self._dphase_dt[k]

                if partials:
                    dw_delem = np.zeros(partials_shape)
                    dw_delem[..., start] = cos_arg
                    dw_delem[..., start+1] = self.amp[k] * cos_arg[k]
                    dw_delem[..., start+2] = dw_delem[..., start+1] * t

                if self._wobbles[k] == 'mean':
                    mean     += w
                    dmean_dt += dw_dt
                    if partials:
                        dmean_delem += dw_delem

                elif self._wobbles[k] == 'peri':
                    peri     += w
                    dperi_dt += dw_dt
                    if partials:
                        dperi_delem += dw_delem

                elif self._wobbles[k] == 'node':
                    node     += w
                    dnode_dt += dw_dt
                    if partials:
                        dnode_delem += dw_delem

                elif self._wobbles[k] == 'a':
                    a     += w
                    da_dt += dw_dt
                    if partials:
                        da_delem += dw_delem

                elif self._wobbles[k] == 'e':
                    e     += w
                    de_dt += dw_dt
                    if partials:
                        de_delem += dw_delem

                else:
                    i     += w
                    di_dt += dw_dt
                    if partials:
                        di_delem += dw_delem

            # Laplace plane case
            else:
                arg = self._phase0[k] + t * self._dphase_dt[k]
                sin_arg = np.sin(arg)
                cos_arg = np.cos(arg)

                laplace_plane = True
                laplace_sin_inc = np.sin(self.amp[k])
                laplace_cos_inc = np.cos(self.amp[k])
                laplace_sin_node = sin_arg
                laplace_cos_node = cos_arg

        ##################################################################################
        # Evaluate some derived elements
        ##################################################################################

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
            dae_delem[..., _SEMIM] = e
            dae_delem[..., _ECCEN] = a

            dcosi_delem = np.zeros(partials_shape)
            dsini_delem = np.zeros(partials_shape)
            dcosi_delem[..., _INCLI] = -sin_i
            dsini_delem[..., _INCLI] =  cos_i

        ##################################################################################
        # Determine moon polar coordinates in orbit plane
        #   r     = a - a * e * cos(mean - peri))
        #   theta = mean -  2 * e * sin(mean - peri)
        ##################################################################################

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
            dcosmp_delem = -sin_mp[..., np.newaxis] * dmp_delem
            dsinmp_delem =  cos_mp[..., np.newaxis] * dmp_delem

            dr_delem = (da_delem - ae[..., np.newaxis] * dcosmp_delem
                        - dae_delem * cos_mp[..., np.newaxis])
            dtheta_delem = (dmean_delem + 2. * (e[..., np.newaxis] * dsinmp_delem
                                                + de_delem * sin_mp[..., np.newaxis]))

        ##################################################################################
        # Locate body on an inclined orbit, in a frame where X is along the
        # ascending node
        #   asc[X] = r cos(theta - node)
        #   asc[Y] = r sin(theta - node) cos(i)
        #   asc[Z] = r sin(theta - node) sin(i)
        ##################################################################################

        tn = theta - node
        cos_tn = np.cos(tn)
        sin_tn = np.sin(tn)

        unit1 = np.empty((time.shape + (3,)))
        unit1[..., 0] = cos_tn
        unit1[..., 1] = sin_tn * cos_i
        unit1[..., 2] = sin_tn * sin_i

        asc = r[..., np.newaxis] * unit1

        # Time-derivatives...
        dtn_dt = dtheta_dt - dnode_dt
        dcostn_dt = -sin_tn * dtn_dt
        dsintn_dt =  cos_tn * dtn_dt

        dunit1_dt = np.empty(time.shape + (3,))
        dunit1_dt[..., 0] = dcostn_dt
        dunit1_dt[..., 1] = dsintn_dt * cos_i + sin_tn * dcosi_dt
        dunit1_dt[..., 2] = dsintn_dt * sin_i + sin_tn * dsini_dt

        dasc_dt = dr_dt[...,np.newaxis] * unit1 + r[...,np.newaxis] * dunit1_dt

        # Partials...
        if partials:
            dtn_delem = dtheta_delem - dnode_delem
            dcostn_delem = -sin_tn[...,np.newaxis] * dtn_delem
            dsintn_delem =  cos_tn[...,np.newaxis] * dtn_delem

            dunit1_delem = np.empty(partials_shape + (3,))
            dunit1_delem[..., 0] = dcostn_delem
            dunit1_delem[..., 1] = (dsintn_delem * cos_i[..., np.newaxis]
                                    + sin_tn[..., np.newaxis] * dcosi_delem)
            dunit1_delem[..., 2] = (dsintn_delem * sin_i[..., np.newaxis]
                                    + sin_tn[..., np.newaxis] * dsini_delem)

            dasc_delem = (dr_delem[..., np.newaxis] * unit1[..., np.newaxis, :]
                          + r[..., np.newaxis, np.newaxis] * dunit1_delem)
            # shape is (..., 9, 3)

        ##################################################################################
        # Rotate the ascending node back into position in our inertial frame
        #   xyz[X] = asc[X] * cos(node) - asc[Y] * sin(node)
        #   xyz[Y] = asc[X] * sin(node) + asc[Y] * cos(node)
        #   xyz[Z] = asc[Z]
        ##################################################################################

        cos_node = np.cos(node)
        sin_node = np.sin(node)

        rotate = np.zeros(time.shape + (3,3))
        rotate[..., 0, 0] =  cos_node
        rotate[..., 0, 1] = -sin_node
        rotate[..., 1, 0] =  sin_node
        rotate[..., 1, 1] =  cos_node
        rotate[..., 2, 2] =  1.

        xyz = np.sum(rotate[..., :, :] * asc[..., np.newaxis, :], axis=-1)

        # Time-derivatives...
        dcosnode_dt = -sin_node * dnode_dt
        dsinnode_dt =  cos_node * dnode_dt

        drotate_dt = np.zeros(time.shape + (3,3))
        drotate_dt[..., 0, 0] =  dcosnode_dt
        drotate_dt[..., 0, 1] = -dsinnode_dt
        drotate_dt[..., 1, 0] =  dsinnode_dt
        drotate_dt[..., 1, 1] =  dcosnode_dt

        dxyz_dt = (np.sum(rotate * dasc_dt[..., np.newaxis, :], axis=-1) +
                   np.sum(drotate_dt * asc[..., np.newaxis, :], axis=-1))

        # Partials...
        if partials:
            dcosnode_delem = -sin_node[..., np.newaxis] * dnode_delem
            dsinnode_delem =  cos_node[..., np.newaxis] * dnode_delem

            drotate_delem = np.zeros(partials_shape + (3, 3))
            drotate_delem[..., 0, 0] =  dcosnode_delem
            drotate_delem[..., 0, 1] = -dsinnode_delem
            drotate_delem[..., 1, 0] =  dsinnode_delem
            drotate_delem[..., 1, 1] =  dcosnode_delem

            dxyz_delem = (np.sum(rotate[..., np.newaxis, :, :]
                                 * dasc_delem[..., np.newaxis, :], axis=-1)
                          + np.sum(drotate_delem
                                   * asc[..., np.newaxis, np.newaxis, :], axis=-1))
            # shape = (..., 9, 3)

        ##################################################################################
        # Apply Laplace Plane
        #   asc[X] = r cos(theta - node)
        #   asc[Y] = r sin(theta - node) cos(i)
        #   asc[Z] = r sin(theta - node) sin(i)
        ##################################################################################

        if laplace_plane:
            node = np.array([laplace_cos_node, laplace_sin_node, 0.])

            cos_sin_node = laplace_cos_node * laplace_sin_node
            cos2_node = laplace_cos_node**2
            sin2_node = laplace_sin_node**2

            cos_sin_node_1_minus_cos_inc = cos_sin_node * (1. - laplace_cos_inc)

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

            xyz = np.sum(rotate[..., :, :] * xyz[..., np.newaxis, :], axis=-1)
            dxyz_dt = np.sum(rotate[..., :, :] * dxyz_dt[..., np.newaxis, :], axis=-1)

            if partials:
                dxyz_delem = np.sum(rotate[..., np.newaxis, :, :]
                                    * dxyz_delem[..., np.newaxis, :], axis=-1)

        ##################################################################################
        # Return results
        ##################################################################################

        pos = Vector3(xyz)
        vel = Vector3(dxyz_dt)

        if partials:
            dxyz_delem = dxyz_delem.swapaxes(-1, -2)
            pos.insert_deriv('elements', Vector3(dxyz_delem, drank=1))

        return (pos, vel)

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, *, quick=None, partials=False):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.
            partials (bool, optional): True to include the derivatives of position with
                respect to the orbital elements.

        Returns:
            (Event): The Event object containing (at least) the time, position, and
                velocity on the Path.
        """

        # Without an observer, return event in the planet frame
        if self.observer is None:
            (pos, vel) = self._xyz_planet(time, partials=partials)
            return Event(time, (pos, vel), self._origin, self._frame)

        # With an observer, return event in J2000, with the light time accounted for
        planet_event = self._photon_from_planet(time, quick=quick)[0]
        (pos, vel) = self._xyz_planet(planet_event.time, partials=partials)
        pos_j2000 = self.to_j2000.rotate(pos) + planet_event.pos
        vel_j2000 = self.to_j2000.rotate(vel) + planet_event.vel
        return Event(time, (pos_j2000, vel_j2000), self._observer, Frame.J2000)

    def _photon_from_planet(self, time, *, derivs=False, guess=None, antimask=None,
                            quick=None, converge={}):

        # Check the cache for the planet event
        events = self._events[time]
        if events is None:
            obs_event = Event(time, Vector3.ZERO, self._observer, self._frame)
            events = self.center.photon_to_event(obs_event, derivs=derivs, guess=guess,
                                                 antimask=antimask, quick=quick,
                                                 converge=converge)
            if np.size(time) == 1:
                self._events[time] = events

        return events

    def node_at_time(self, time):
        """The longitude of ascending node at the specified time.

        Wobbles are ignored. The angle is a positive rotation about the planet's ring
        frame.
        """

        time = Scalar.as_scalar(time)
        return self._elements[_NODE0] + (time - self.epoch) * self._elements[_DNODE]

    def pole_at_time(self, time):
        """The J2000 vector pointing toward the orbit's pole at the specified time.

        Wobbles are ignored.
        """

        xform = self._frame_wrt_j2000.transform_at_time(time)
        x_axis_in_j2000 = xform.unrotate(Vector3.XAXIS)
        y_axis_in_j2000 = xform.unrotate(Vector3.YAXIS)
        z_axis_in_j2000 = xform.unrotate(Vector3.ZAXIS)

        node = self.node_at_time(time)
        cos_node = np.cos(node)
        sin_node = np.sin(node)

        # This vector is 90 degrees behind of the node in the reference equator
        target_in_j2000 = (sin_node * x_axis_in_j2000 - cos_node * y_axis_in_j2000)

        return self.cos_i * z_axis_in_j2000 + self.sin_i * target_in_j2000

    ######################################################################################
    # Override for the case where observer != None
    ######################################################################################

    def photon_to_event(self, arrival, derivs=False, *, guess=None, antimask=None,
                        quick=None, converge=None, partials=False):
        """The photon departure event from this path to match the arrival event.

        This is an override of the default method, provided to support the partial
        derivatives.
        """

        if self._observer is None:
            (path_event,
             obs_event) = Path.photon_to_event(self, arrival, derivs=derivs, guess=guess,
                                               antimask=antimask, quick=quick,
                                               converge=converge)
            if partials:
                (pos, vel) = self._xyz_planet(path_event.time, partials=True)
                path_event.pos.insert_deriv('elements', pos.d_delements)

            return (path_event, obs_event)

        (planet_event,
         obs_event) = self._photon_from_planet(arrival.time, derivs=derivs, guess=guess,
                                               antimask=antimask, quick=quick,
                                               converge=converge)

        path_event = self.event_at_time(planet_event.time, quick=quick, partials=partials)
        path_event.dep_lt = path_event.time - obs_event.time
        path_event.dep_j2000 = path_event.pos_j2000 - obs_event.pos_j2000

        obs_event.arr_lt = path_event.dep_lt
        obs_event.arr_j2000 = path_event.dep_j2000

        return (path_event, obs_event)

##########################################################################################

Path._PATH_SUBCLASSES.append(KeplerPath)

##########################################################################################
