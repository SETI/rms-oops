##########################################################################################
# oops/surface/ansa.py: Ansa subclass of class Surface
##########################################################################################

import numpy as np

from polymath               import Scalar, Vector3
from oops.constants         import TWOPI
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.surface.surface_  import Surface
from oops.surface.ringplane import RingPlane


class Ansa(Surface):
    """The locus of points where a radius vector from the pole of the Z-axis is
    perpendicular to the line of sight.

    This provides a convenient coordinate system for describing rings when viewed nearly
    edge-on. The coordinates are `(r, z, theta)`, where:

    * `r` (Scalar): Radial distance from the Z-axis, positive on the "right" side if Z
      points "up", negative on the left side.
    * `z` (Scalar): Vertical distance from the (x,y) plane.
    * `theta` (Scalar): Angular distance from the ansa, positive further away from the
      observer and negative closer.
    """

    COORDINATE_TYPE = 'cylindrical'
    COORDINATE_NAMES = ('radius', 'elevation', 'longitude')
    COORDINATE_ABBREVS = ('r', 'z', 'theta')
    COORDINATE_RANGES = ((0, None), (None, None), (0, TWOPI))
    IS_VIRTUAL = True

    def __init__(self, origin, frame, *, gravity=None, ringplane=None, radii=None):
        """Constructor for an Ansa Surface.

        Parameters:
            origin (Path or str): The Path or the ID of the Path defining the motion of
                the center of the ring system.
            frame (Frame or str): The Frame or the ID of the Frame in which the ring plane
                is the (x,y) plane (where z == 0).
            gravity (Gravity, optional): The Gravity object used to define the orbital
                velocities relative to the Surface.
            ringplane (RingPlane, optional): A RingPlane associated with this Ansa
                Surface. If provided, this Surface inherits the gravity field and radial
                limits of the RingPlane, unless they are given as input.
            radii (tuple[float, float], optional): The nominal inner and outer radii of
                the ring, in km; None for a ring with no radial limits.
        """

        self.origin  = Path.as_waypoint(origin)
        self.frame   = Frame.as_wayframe(frame)

        if radii is None:
            self._radii = None
        else:
            self._radii = np.asarray(radii, dtype=np.float64)

        self._state_ringplane = ringplane
        if ringplane is None:
            self._ringplane = RingPlane(self.origin, self.frame, radii=radii,
                                       gravity=gravity)
        else:
            self._ringplane = ringplane

        if gravity is None:
            self._gravity = self._ringplane._gravity
        else:
            self._gravity = gravity

        # Save the unmasked version of this surface
        if self._radii is None:
            self.unmasked = self
        else:
            self.unmasked = Ansa(self.origin, self.frame, gravity=self._gravity,
                                 ringplane=self._ringplane, radii=None)

        # Unique key for intercept calculations
        self.intercept_key = ('ansa', self.origin.waypoint, self.frame.wayframe)

    def __getstate__(self):
        self.refresh()
        return (Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame),
                self._gravity, self._state_ringplane,
                None if self._radii is None else tuple(self._radii))

    def __setstate__(self, state):
        (origin, frame, gravity, ringplane, radii) = state
        self.__init__(origin, frame, gravity=gravity, ringplane=ringplane, radii=radii)
        self.freeze()

    @staticmethod
    def for_ringplane(ringplane):
        """Construct an Ansa Surface associated with a given RingPlane, ignoring any
        modes.

        Parameters:
            ringplane (RingPlane): The RingPlane relative to which this Ansa Surface is
                to be defined.

        Returns:
            Ansa: The Ansa Surface sharing the origin, frame, gravity, and radii of the
            given RingPlane.
        """

        return Ansa(ringplane.origin, ringplane.frame, gravity=ringplane._gravity,
                    ringplane=ringplane, radii=ringplane._radii)

    @staticmethod
    def for_body(body):
        """Construct an Ansa Surface associated with a given body, ignoring any modes.

        Parameters:
            body (Body): The ring Body relative to which this Ansa Surface is to be
                defined. If the Body's surface is not a ring plane, its `ring_body` is
                used instead.

        Returns:
            Ansa: The Ansa Surface sharing the path, frame, gravity, and radii of the ring
            Body.
        """

        # Identify the ring body
        if body.surface.COORDINATE_TYPE != 'polar':
            body = body.ring_body

        return Ansa(body.path, body.frame, gravity=body.gravity,
                    ringplane=body.surface, radii=body.surface._radii)

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                            hints=None):
        """Surface coordinates associated with a position vector.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame.
            time (Scalar, optional): Time at which to evaluate the Surface.
            axes (int, optional): 2 or 3, indicating whether to return the first two
                coordinates (rad, z) or all three (rad, z, theta) as Scalars.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            tuple[Scalar, ...]: Two or three coordinate values, optionally followed by
            `hints`, where:

            * `rad` (Scalar): Projected distance from the body pole, in km.
            * `z` (Scalar): Projected vertical distance above the ring plane, in km.
            * `theta` (Scalar): Longitude of the intercept point, in radians; included if
              `axes` == 3.
            * `hints` (Any): The input value of `hints`, included if it is not None.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)

        pos = Vector3.as_vector3(pos, recursive=derivs)
        obs = Vector3.as_vector3(obs, recursive=derivs)
        (pos_x, pos_y, pos_z) = pos.to_scalars()
        (obs_x, obs_y, obs_z) = obs.to_scalars()

        rabs   = (pos_x**2 + pos_y**2).sqrt()
        obs_xy = (obs_x**2 + obs_y**2).sqrt()

        # Find the longitude of pos relative to obs
        lon = pos_y.arctan2(pos_x) - obs_y.arctan2(obs_x)

        # Put it in the range -pi to pi
        lon = ((lon + Scalar.PI) % Scalar.TWOPI) - Scalar.PI
        sign = lon.sign()
        r = rabs * sign

        # Apply mask as needed
        if self._radii is not None:
            mask = r.tvl_lt(self._radii[0]) | r.tvl_gt(self._radii[1])
            if mask.any_true_or_masked():
                r = r.remask_or(mask.vals)
                pos_z = pos_z.remask(r.mask)

        # Fill in the third coordinate if necessary
        if axes > 2:
            # As discussed in the math found below with vector3_from_coords(),
            # the ansa longitude relative to the observer is:

            phi = (rabs / obs_xy).arccos()
            theta = sign*lon - phi
            if self._radii is not None:
                theta = theta.remask(r.mask)

            results = (r, pos_z, theta)
        else:
            results = (r, pos_z)

        if hints is not None:
            results += (hints,)

        return results

    def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False,
                            hints=None):
        """The position where a point with the given coordinates falls relative to this
        surface's origin and frame.

        Parameters:
            coords (tuple[Scalar, ...]): Two or three Scalars defining coordinates at
                or near this surface. These can have different shapes, but must be
                broadcastable to a common shape.

                * `rad` (km): Projected distance from the body pole.
                * `z` (km): Projected vertical distance above the ring plane.
                * `theta` (rad, optional): Longitude of the intercept point.

            obs (Vector3): Observer position relative to this Surface's origin and frame.
                Required, because this is a virtual surface.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored by
                this Surface subclass.
            derivs (bool, optional): True to propagate any derivatives inside the
                coordinates and obs into the returned position vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: Points defined by the coordinates, relative to
            this Surface's origin and frame, optionally followed by `hints`. The input
            value of `hints` is returned if it is not None.
        """

        # Validate inputs
        self._vector3_from_coords_check(coords)

        # Given (r,z, theta) and the observer position, solve for position.
        #   pos = (|r| cos(a), |r| sin(a), z)
        # where angle a is defined by the location of the observer.
        #
        # theta = 0 at the ansa, where los and pos are perpendicular.
        # theta < 0 for points closer along the los, > 0 for points further.
        #
        # First solve for a where theta = 0.
        #
        #   pos_xy dot (obs_xy - pos_xy) = 0
        #   pos_xy dot pos_xy = pos_xy dot obs_xy
        #   r**2 = |r| cos(a) obs_x + |r| sin(a) obs_y
        #
        # For convenience, define the coordinate system so that obs falls on the
        # (x,z) plane, so obs_y = 0 and obs_x > 0.
        #
        #   r**2 = |r| obs_x cos(a)
        #
        #   cos(a) = |r| / obs_x
        #
        #   a = sign * arccos(|r| / obs_x)
        #
        # Define phi as the arccos term:
        #
        #   a = sign * phi(r,obs_x)
        #
        # Two solutions exist, symmetric about the (x,z) plane, as expected. The
        # positive sign corresponds to ring longitudes ahead of the observer,
        # which we define as the "right" ansa. The negative sign identifies the
        # "left" ansa.
        #
        # Theta is an angular offset from phi, with smaller values closer to the
        # observer and larger angles further away.

        r = Scalar.as_scalar(coords[0], recursive=derivs)
        z = Scalar.as_scalar(coords[1], recursive=derivs)

        sign = r.sign()
        rabs = r * sign

        if len(coords) == 2:
            theta = Scalar(0.)
        else:
            theta = Scalar.as_scalar(coords[2], recursive=derivs)

        (obs_x, obs_y, obs_z) = Vector3.as_vector(obs, recursive=derivs).to_scalars()
        obs_xy = (obs_x**2 + obs_y**2).sqrt()

        phi = (rabs / obs_xy).arccos()

        pos_lon = obs_y.arctan2(obs_x) + sign * (phi + theta)

        pos = Vector3.from_scalars(rabs * pos_lon.cos(),
                                   rabs * pos_lon.sin(), z)

        if hints is not None:
            return (pos, hints)

        return pos

    def intercept(self, obs, los, *, time=None, direction='dep', derivs=False, guess=None,
                  hints=None):
        """The position where a specified line of sight intercepts the Surface.

        Parameters:
            obs (Vector3): Observer position as a Vector3 relative to this Surface's
                origin and frame.
            los (Vector3): Line of sight as a Vector3 in this Surface's frame.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored by
                this Surface subclass.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface; ignored here.
            derivs (bool, optional): True to propagate any derivatives inside obs and los
                into the returned intercept point.
            guess (Scalar, optional): Unused.
            hints (Any, optional): If not None (the default), this value is appended to
                the returned tuple. Needed for compatibility with other Surface
                subclasses.

        Returns:
            tuple[Vector3, Scalar[, Any]]: `(pos, t)` or `(pos, t, hints)`, where:

            * `pos` (Vector3): Intercept points on the Surface relative to this surface's
              origin and frame, in km.
            * `t` (Scalar): Value such that `intercept = obs + t * los`.
            * `hints` (Any): The input value of `hints`, included if it is not None.
        """

        obs = Vector3.as_vector3(obs, recursive=derivs)
        los = Vector3.as_vector3(los, recursive=derivs)

        # (obs_xy + t los_xy) dot los_xy = 0
        # t = -(obs_xy dot los_xy) / (los_xy dot los_xy)
        # pos = obs + t * los

        obs_x = obs.to_scalar(0)
        obs_y = obs.to_scalar(1)
        los_x = los.to_scalar(0)
        los_y = los.to_scalar(1)

        los_sq = los_x**2 + los_y**2

        obs_dot_los = obs_x * los_x + obs_y * los_y
        t = -obs_dot_los / los_sq

        pos = obs + t * los

        if hints is not None:
            return (pos, t, hints)

        return (pos, t)

    def normal(self, pos, *, obs=None, time=None, derivs=False, hints=None):
        """The normal vector at a position at or near a surface.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored when determining the normal to an ansa.
            time (Scalar, optional): Time at which to evaluate the Surface.
            derivs (bool, optional): True to propagate any derivatives of pos into the
                returned normal vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: Directions normal to the Surface that pass
            through the position, optionally followed by `hints`. Vector lengths are
            arbitrary, and the input value of `hints` is returned if it is not None.

        Notes:
            The ansa normal is defined as the ring plane normal, so that incidence and
            emission angles match those of the associated ring plane.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # Always the Z-axis
        perp = pos.as_all_constant((0.,0.,1.))

        if hints is not None:
            return (perp, hints)

        return perp

##########################################################################################
