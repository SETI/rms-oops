################################################################################
# oops/surface/ansa.py: Ansa subclass of class Surface
################################################################################

import numpy as np

from polymath               import Scalar, Vector3
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.surface.surface_  import Surface
from oops.surface.ringplane import RingPlane

class Ansa(Surface):
    """This surface is defined as the locus of points where a radius vector from
    the pole of the Z-axis is perpendicular to the line of sight. This provides
    a convenient coordinate system for describing rings when viewed nearly
    edge-on. The coordinates are (r,z,theta) where
        r       radial distance from the Z-axis, positive on the "right" side
                (if Z is pointing "up"); negative on the left side.
        z       vertical distance from the (x,y) plane.
        theta   angular distance from the ansa, with positive values further
                away from the observer and negative values closer.
    """

    COORDINATE_TYPE = 'cylindrical'
    IS_VIRTUAL = True

    #===========================================================================
    def __init__(self, origin, frame, gravity=None, ringplane=None, radii=None):
        """Constructor for an Ansa Surface.

        Input:
            origin      a Path object or ID defining the motion of the center
                        of the ring system.

            frame       a Frame object or ID in which the ring plane is the
                        (x,y) plane (where z == 0).

            gravity     an optional Gravity object, used to define the orbital
                        velocities relative to the surface.

            ringplane   an optional RingPlane object associated with this Ansa
                        surface. If provided, this surface inherits the gravity
                        field and radial limits of the RingPlane, unless they
                        are given as input.

            radii       the nominal inner and outer radii of the ring, in km.
                        None for a ring with no radial limits.
        """

        self.origin  = Path.as_waypoint(origin)
        self.frame   = Frame.as_wayframe(frame)

        if radii is None:
            self.radii = None
        else:
            self.radii = np.asarray(radii, dtype=np.float64)

        self._state_ringplane = ringplane
        if ringplane is None:
            self.ringplane = RingPlane(self.origin, self.frame, radii=radii,
                                       gravity=gravity)
        else:
            self.ringplane = ringplane

        if gravity is None:
            self.gravity = self.ringplane.gravity
        else:
            self.gravity = gravity

        # Save the unmasked version of this surface
        if self.radii is None:
            self.unmasked = self
        else:
            self.unmasked = Ansa(self.origin, self.frame,
                                 gravity=self.gravity,
                                 ringplane=self.ringplane,
                                 radii=None)

        # Unique key for intercept calculations
        self.intercept_key = ('ansa', self.origin.waypoint,
                                      self.frame.wayframe)

    def __getstate__(self):
        return (Path.as_primary_path(self.origin),
                Frame.as_primary_frame(self.frame),
                self.gravity, self._state_ringplane, tuple(self.radii))

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def for_ringplane(ringplane):
        """Construct an Ansa Surface associated with a given RingPlane, ignoring
        any modes.

        Input:
            ringplane   a ringplane surface relative to which this ansa surface
                        is to be defined.
        """

        return Ansa(ringplane.origin, ringplane.frame, gravity=ringplane.gravity,
                    ringplane=ringplane, radii=ringplane.radii)

    #===========================================================================
    @staticmethod
    def for_body(body):
        """Construct an Ansa Surface associated with a given body, ignoring any
        modes.

        Input:
            body        a ring body to which this ansa surface is to be defined.
        """

        # Identify the ring body
        if body.surface.COORDINATE_TYPE != 'polar':
            body = body.ring_body

        return Ansa(body.path, body.frame, gravity=body.gravity,
                    ringplane=body.surface, radii=body.surface.radii)

    #===========================================================================
    def coords_from_vector3(self, pos, obs=None, time=None, axes=2,
                                  derivs=False, hints=None):
        """Surface coordinates associated with a position vector.

        Input:
            pos         a Vector3 of positions at or near the surface, relative
                        to this surface's origin and frame.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame.
            time        a Scalar time at which to evaluate the surface.
            axes        2 or 3, indicating whether to return the first two
                        coordinates (rad, z) or all three (rad, z, theta) as
                        Scalars.
            derivs      True to propagate any derivatives inside pos and obs
                        into the returned coordinates.
            hints       ignored. Provided for compatibility with other Surface
                        subclasses.

        Return:         coordinate values packaged as a tuple containing two or
                        three Scalars, one for each coordinate.
            rad         projected distance from the body pole, in km.
            z           projected vertical distance above the ring plane, in km.
            theta       longitude of the intercept point, in radians; included
                        if axes == 3.
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
        if self.radii is not None:
            mask = r.tvl_lt(self.radii[0]) | r.tvl_gt(self.radii[1])
            if mask.any():
                r = r.remask_or(mask.vals)
                pos_z = pos_z.remask(r.mask)

        # Fill in the third coordinate if necessary
        if axes > 2:
            # As discussed in the math found below with vector3_from_coords(),
            # the ansa longitude relative to the observer is:

            phi = (rabs / obs_xy).arccos()
            theta = sign*lon - phi
            if self.radii is not None:
                theta.remask(r.mask)

            return (r, pos_z, theta)

        return (r, pos_z)

    #===========================================================================
    def vector3_from_coords(self, coords, obs, time=None, derivs=False):
        """The position where a point with the given coordinates falls relative
        to this surface's origin and frame.

        Input:
            coords      a tuple of two or three Scalars defining coordinates at
                        or near this surface. These can have different shapes,
                        but must be broadcastable to a common shape.
                rad     projected distance in km from the body pole.
                z       projected vertical distance in km above the ring plane.
                theta   longitude in radians of the intercept point.
            obs         a Vector3 of observer position relative to this
                        surface's origin and frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        by this Surface subclass.
            derivs      True to propagate any derivatives inside the coordinates
                        and obs into the returned position vectors.

        Return:         a Vector3 of points defined by the coordinates, relative
                        to this surface's origin and frame.
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
        return pos

    #===========================================================================
    def intercept(self, obs, los, time=None, direction='dep', derivs=False,
                                  guess=None, hints=None):
        """The position where a specified line of sight intercepts the surface.

        Input:
            obs         observer position as a Vector3 relative to this
                        surface's origin and frame.
            los         line of sight as a Vector3 in this surface's frame.
            time        a Scalar time at which to evaluate the surface; ignored
                        by this Surface subclass.
            direction   'arr' for a photon arriving at the surface; 'dep' for a
                        photon departing from the surface; ignored here.
            derivs      True to propagate any derivatives inside obs and los
                        into the returned intercept point.
            guess       unused.
            hints       if not None (the default), this value is appended to the
                        returned tuple. Needed for compatibility with other
                        Surface subclasses.

        Return:         a tuple (pos, t) or (pos, t, hints), where
            pos         a Vector3 of intercept points on the surface relative
                        to this surface's origin and frame, in km.
            t           a Scalar such that:
                            intercept = obs + t * los
            hints       the input value of hints, included if it is not None.
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

    #===========================================================================
    def normal(self, pos, time=None, derivs=False):
        """The normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface relative
                        to this surface's origin and frame.
            time        a Scalar time at which to evaluate the surface.
            derivs      True to propagate any derivatives of pos into the
                        returned normal vectors.

        Return:         a Vector3 containing directions normal to the surface
                        that pass through the position. Lengths are arbitrary.

        NOTE: We define this as the ansa normal as the ring plane normal so that
        incidence and emission angles are the same as those for the associated
        ring plane.
        """

        pos = Vector3.as_vector3(pos, recursive=derivs)

        # Always the Z-axis
        return pos.as_all_constant((0.,0.,1.))

################################################################################
