##########################################################################################
# oops/surface/ringplane.py
##########################################################################################

import numpy as np

from polymath                   import Scalar, Vector3
from oops.constants             import TWOPI
from oops.frame.frame_          import Frame
from oops.path.path_            import Path
from oops.surface.surface_      import Surface
from oops.gravity.oblategravity import OblateGravity


class RingPlane(Surface):
    """A flat surface in the *(x,y)* plane with optional Keplerian motion.

    A subclass of :class:`~oops.Surface` in which the optional velocity field is defined
    by circular Keplerian motion about the center point. Coordinates are cylindrical
    (radius, longitude, elevation), with an optional offset in elevation from the
    equatorial (z=0) plane.

    Optional modes can be used to apply sinusoidal offset patterns in the radial
    coordinate.
    """

    COORDINATE_TYPE = 'polar'
    COORDINATE_NAMES = ('radius', 'longitude', 'elevation')
    COORDINATE_ABBREVS = ('r', 'theta', 'z')
    COORDINATE_RANGES = ((0, None), (0, TWOPI), (None, None))
    IS_VIRTUAL = False
    IS_TIME_DEPENDENT = False

    def __init__(self, origin, frame, *, radii=None, gravity=None, elevation=0.,
                 modes=None, epoch=0.):
        """Constructor for a RingPlane surface.

        Parameters:
            origin (Path or str): The Path or the ID of the Path defining the motion of
                the center of the ring plane.
            frame (Frame or str): The Frame or the ID of the Frame in which the ring
                plane is the *(x,y)* plane (where ``z = 0``).
            radii (tuple[float, float], optional): The nominal inner and outer radii of
                the ring, in km; None for a ring with no radial limits.
            gravity (Gravity, optional): Gravity model of the central body, used to define
                the orbital velocities within the plane.
            elevation (Scalar, optional): A possible offset of the ring plane in the
                direction of positive rotation, in km.
            modes (list, optional): Zero or more radial modes in the ring. Each mode is
                described by a tuple of four parameters `(cycles, amp, peri0, speed)`:

                * `cycles` (int): The number of radial cycles around the ring.
                * `amp` (float): The radial amplitude in km.
                * `peri0` (float): The longitude of one radial minimum at `epoch` in
                  radians.
                * speed (float): The pattern speed in radians per second.

            epoch (float, optional): The epoch at which the radial mode parameters apply.
                Not used unless radial modes are present.
        """

        self.origin    = Path.as_waypoint(origin)
        self.frame     = Frame.as_wayframe(frame)
        self._gravity   = gravity
        self._elevation = float(elevation)
        self._modes     = modes or []
        self._epoch     = float(epoch)
        self.IS_TIME_DEPENDENT = bool(modes)

        if radii is None:
            self._radii = None
        else:
            self._radii    = np.asarray(radii, dtype=np.float64)
            self._radii_sq = self._radii**2

        # Save the unmasked version of this surface
        if radii is None:
            self.unmasked = self
        else:
            self.unmasked = RingPlane(self.origin, self.frame,
                                      radii = None,
                                      gravity = self._gravity,
                                      elevation = self._elevation,
                                      modes = self._modes,
                                      epoch = self._epoch)

        # Identify the maximum orbital rate by any means necessary; without this limit,
        # speeds near the origin get ridiculous. Without gravity, there is no velocity
        # field, so no rate limit is needed.
        if self._gravity is None:
            self._max_rate = None
        elif self._radii is not None:
            r = self._radii[0]
            self._max_rate = self._gravity.n(r)
        elif hasattr(self._gravity, 'rp'):
            r = self._gravity.rp
            self._max_rate = self._gravity.n(r)
        else:
            # If we can't figure out the planet, clamp the rate at that for an orbit
            # skimming the surface of Neptune. (Note that this rate is faster than that
            # for Jupiter, Saturn, or Uranus.)
            neptune = OblateGravity.NEPTUNE
            self._max_rate = neptune.n(neptune.rp)

        # Unique key for intercept calculations:
        # ('ring', origin, frame, elevation, i, node, dnode_dt, epoch)
        # Extra elements are so OrbitPlane and RingPlane can share the same key in
        # situations where the orbit is not inclined.
        self.intercept_key = ('ring', self.origin.waypoint, self.frame.wayframe,
                              self._elevation, 0., 0., 0., 0.)

    def __getstate__(self):
        self.refresh()
        kwargs = {'radii': None if self._radii is None else tuple(self._radii),
                  'gravity': self._gravity,
                  'elevation': self._elevation,
                  'modes': self._modes,
                  'epoch': self._epoch}
        return (Path.as_primary_path(self.origin), Frame.as_primary_frame(self.frame),
                kwargs)

    def __setstate__(self, state):
        (origin, frame, kwargs) = state
        self.__init__(origin, frame, **kwargs)
        self.freeze()

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                            hints=None):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored unless
                this RingPlane contains radial modes, in which case it is required.
            axes (int, optional): 2 or 3, indicating whether to return the first two
                coordinates (rad, theta) or all three (rad, theta, z) as Scalars.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            tuple[Scalar, ...]: Two or three coordinate values, optionally followed by
            `hints`, where:

            * `rad` (Scalar): Mean orbital radius in the ring plane, in km.
            * `theta` (Scalar): Longitude in radians of the intercept point.
            * `z` (Scalar): Vertical distance in km above the ring plane; included if
              `axes` == 3.
            * `hints` (Any): The input value of `hints`, included if it is not None.

        Raises:
            ValueError: If this RingPlane contains radial modes and no `time` is given.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)
        pos = Vector3.as_vector3(pos, recursive=derivs)

        # Generate cylindrical coordinates
        (r, theta, z) = pos.to_cylindrical()

        if self._modes:
            a = r - self._mode_offset(theta, time, derivs=derivs)
        else:
            a = r

        # Apply mask as needed
        if self._radii is not None:
            mask = a.tvl_lt(self._radii[0]) | a.tvl_gt(self._radii[1])
            if mask.any_true_or_masked():       # this allows for fully masked results
                a = a.remask_or(mask.vals)
                theta = theta.remask(a.mask)
                if axes > 2:
                    z = z.remask(a.mask)

        if axes == 2:
            results = (a, theta)
        elif self._elevation == 0:
            results = (a, theta, z)
        else:
            results = (a, theta, z - self._elevation)

        if hints is not None:
            results += (hints,)

        return results

    def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False,
                            hints=None):
        """The position at the given surface coordinates.

        Parameters:
            coords (tuple[Scalar, ...]): Two or three Scalars defining coordinates at or
                near this surface. These can have different shapes, but must be
                broadcastable to a common shape.

                * `r`: The mean orbital radius in the ring plane, in km.
                * `theta`: The longitude in radians of the selected point.
                * `z`: Vertical distance in km above the ring plane, optional.

            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored unless
                this RingPlane contains radial modes, in which case it is required.
            derivs (bool, optional): True to propagate any derivatives inside `coords` and
                `obs` into the returned position vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this subclass. If it is not None, its value is
                appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: Points defined by the coordinates, relative to
            this Surface's origin and frame, optionally followed by `hints`. The input
            value of `hints` is returned if it is not None.

        Raises:
            ValueError: If this RingPlane contains radial modes and no `time` is given.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)
        a = Scalar.as_scalar(coords[0], recursive=derivs)
        theta = Scalar.as_scalar(coords[1], recursive=derivs)

        if self._modes:
            r = a + self._mode_offset(theta, time, derivs=derivs)
        else:
            r = a

        if len(coords) > 2:
            z = Scalar.as_scalar(coords[2] + self._elevation, recursive=derivs)
        else:
            z = Scalar.as_scalar(self._elevation, recursive=derivs)

        x = r * theta.cos()
        y = r * theta.sin()
        pos = Vector3.from_scalars(x, y, z)

        if hints is not None:
            return (pos, hints)

        return pos

    def intercept(self, obs, los, *, time=None, direction='dep', derivs=False, guess=None,
                  hints=None):
        """The position where a specified line of sight intercepts this Surface.

        Parameters:
            obs (Vector3): Observer position as a Vector3 relative to this Surface's
                origin and frame.
            los (Vector3): Line of sight as a Vector3 in this Surface's frame.
            time (Scalar, optional): Time at the surface; ignored here.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface; ignored here.
            derivs (bool, optional): True to propagate any derivatives inside obs and los
                into the returned intercept point.
            guess (Scalar, optional): Unused.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            tuple[Vector3, Scalar[, Any]]: `(pos, t)` or `(pos, t, hints)`, where:

            * `pos` (Vector3): Intercept points on the Surface relative to this surface's
              origin and frame, in km.
            * `t` (Scalar): Such that ``intercept = obs + t * los``.
            * `hints` (Any): The input value of `hints`, included if it is not None.
        """

        # Solve for obs + factor * los for scalar t, such that the z-component equals
        # zero.
        obs = Vector3.as_vector3(obs, recursive=derivs)
        los = Vector3.as_vector3(los, recursive=derivs)

        obs_z = obs.to_scalar(2)
        los_z = los.to_scalar(2)

        t = (self._elevation - obs_z)/los_z
        pos = obs + t * los

        # Mask based on radial limits if necessary
        if self._radii is not None:
            r_sq = pos.norm_sq(recursive=derivs)
            mask = (r_sq < self._radii_sq[0]) | (r_sq > self._radii_sq[1])
            if np.any(mask):
                pos = pos.remask_or(mask)
            t = t.remask(pos.mask)

        if hints is not None:
            return (pos, t, hints)

        return (pos, t)

    def normal(self, pos, *, obs=None, time=None, derivs=False, hints=None):
        """The normal vector at a position at or near a surface.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored here.
            derivs (bool, optional): True to propagate any derivatives of `pos` into the
                returned normal vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: Directions normal to the Surface that pass
            through the position, optionally followed by `hints`. Vector lengths are
            arbitrary, and the input value of `hints` is returned if it is not None.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # Always the Z-axis
        perp = pos.as_all_constant((0.,0.,1.))

        # The normal is undefined outside the ring's radial limits
        if self._radii is not None:
            r_sq = pos.norm_sq(recursive=derivs)
            mask = (r_sq < self._radii_sq[0]) | (r_sq > self._radii_sq[1])
            if np.any(mask):
                perp = perp.remask_or(mask)

        if hints is not None:
            return (perp, hints)

        return perp

    def velocity(self, pos, *, obs=None, time=None):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or local wind
        speeds on a planet.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored unless
                this RingPlane contains radial modes, in which case it is required.

        Returns:
            Vector3: Velocities, in units of km/s.

        Raises:
            ValueError: If this RingPlane contains radial modes and no `time` is given.
        """

        pos = Vector3.as_vector3(pos, recursive=False)

        # Handle special case that's easy
        if self._gravity is None and not self._modes:
            return Vector3.zeros(pos.shape, mask=pos.mask)

        # Generate info about intercept points
        (x,y,z) = pos.to_scalars(recursive=False)
        radius = (x**2 + y**2).sqrt()
        r_vector = Vector3.from_scalars(x,y,0.)

        # Handle radial modes
        if self._modes:
            lon = y.arctan2(x)
            (offset, dr_dt, dlon_dt) = self._mode_offset(lon, time, rates=True)
            a = radius - offset

            if self._gravity:
                dlon_dt += Scalar.minimum(self._gravity.n(a.vals), self._max_rate)

            v_radial = (dr_dt / radius) * r_vector
            v_angular = dlon_dt * Vector3.ZAXIS.cross(r_vector)
            vflat = v_radial + v_angular

        # Handle simple gravity
        else:
            a = radius
            n = Scalar.minimum(self._gravity.n(a.vals), self._max_rate)
            vflat = n * Vector3.ZAXIS.cross(r_vector)

        # The velocity is undefined outside the ring's radial limits
        if self._radii is not None:
            mask = (a < self._radii[0]) | (a > self._radii[1])
            if np.any(mask):
                vflat = vflat.remask_or(mask)

        return vflat

    ######################################################################################
    # Radius conversions
    ######################################################################################

    def _mode_offset(self, lon, time, *, derivs=False, rates=False):
        """The sum of the modes as a radial offset from the mean epicyclic radius.

        Parameters:
            lon (Scalar): Longitude in radians of the intercept point.
            time (Scalar): Time at which to evaluate the modes, in seconds TDB. It is
                required, because the modes vary with it.
            derivs (bool, optional): True to propagate any derivatives of `lon` and `time`
                into the returned offset.
            rates (bool, optional): True to return the epicyclic rates along with the
                radial offset.

        Returns:
            Scalar or tuple[Scalar, Scalar, Scalar]: The radial offset in km, or
            `(offset, dr_dt, dlon_dt)` if `rates` is True.

        Raises:
            ValueError: If this RingPlane contains radial modes and no `time` is given.
        """

        # The modes vary with time, so there is no sensible value to assume for one that
        # was not given.
        if self._modes and time is None:
            raise ValueError(f'{type(self).__name__} with radial modes requires a time')

        offset = 0.
        dr_dt = 0.
        dlon_dt = 0.
        for mode in self._modes:
            (cycles, amp, peri0, speed) = mode
            arg = cycles * (lon - peri0) + speed * (time - self._epoch)
            amp_cos_arg = amp * arg.cos(recursive=derivs)
            offset = offset - amp_cos_arg
            if rates:
                dr_dt   = dr_dt   + (speed * amp) * arg.sin()
                dlon_dt = dlon_dt + (2. * speed) * amp_cos_arg

        if rates:
            return (offset, dr_dt, dlon_dt)

        return offset

##########################################################################################
