##########################################################################################
# oops/surface/nullsurface.py
##########################################################################################

from polymath import Scalar, Vector3
from oops.frame.frame_     import Frame
from oops.path.path_       import Path
from oops.surface.surface_ import Surface


class NullSurface(Surface):
    """An infinitesimal surface centered on a path and using a given frame.

    A subclass of :class:`~oops.Surface` whose coordinates are the rectangular
    coordinates `(x,y,z)` relative to its origin and frame.
    """

    COORDINATE_TYPE = 'rectangular'
    COORDINATE_NAMES = ('x', 'y', 'z')
    COORDINATE_ABBREVS = ('x', 'y', 'z')
    COORDINATE_RANGES = ((None, None), (None, None), (None, None))

    def __init__(self, origin, frame):
        """Constructor for a NullSurface surface.

        Parameters:
            origin (Path or str): The Path or the ID of the Path defining the motion of
                the center of the ring plane.
            frame (Frame or str): The Frame or the ID of the Frame in which this
                Surface's "normal" is defined by the z-axis.
        """

        self.origin = Path.as_waypoint(origin)
        self.frame  = Frame.as_wayframe(frame)
        self.unmasked = self

        # Unique key for intercept calculations
        self.intercept_key = ('null', self.origin.waypoint, self.frame.wayframe)

    def __getstate__(self):
        self.refresh()
        return (Path.as_primary_path(self.origin), Frame.as_primary_frame(self.frame))

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                            hints=None):
        """Surface coordinates associated with a position vector.

        For NullSurface, the coordinates are simply the (x,y,z) rectangular coordinates
        relative to the surface's origin and frame.

        Parameters:
            pos (Vector3): Positions at or near the Surface, relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            axes (int, optional): 2 or 3, indicating whether to return the first two
                coordinates (x, y) or all three (x, y, z) coordinates as Scalars.
            derivs (bool, optional): True to propagate any derivatives inside pos and obs
                into the returned coordinates.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            tuple[Scalar, ...]: Two or three coordinate values, one for each coordinate,
            optionally followed by `hints`. The input value of `hints` is returned if it
            is not None.
        """

        # Validate inputs
        self._coords_from_vector3_check(axes)

        # Simple rectangular coordinates
        pos = Vector3.as_vector3(pos, recursive=derivs)
        results = pos.to_scalars(recursive=derivs)[:axes]

        if hints is not None:
            results += (hints,)

        return results

    def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False,
                            hints=None):
        """The position at the given surface coordinates.

        Parameters:
            coords (tuple): Two or three Scalars defining coordinates at or near this
                surface. These are the (x,y,z) rectangular coordinates relative to the
                surface's origin and frame. They can have different shapes, but must be
                broadcastable to a common shape.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
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

        # Convert to Scalars
        x = Scalar.as_scalar(coords[0], recursive=derivs)
        y = Scalar.as_scalar(coords[1], recursive=derivs)

        if len(coords) == 2:
            z = Scalar(0.)
        else:
            z = Scalar.as_scalar(coords[2], recursive=derivs)

        # Convert to a Vector3 and return
        pos = Vector3.from_scalars(x, y, z)

        if hints is not None:
            return (pos, hints)

        return pos

    def intercept(self, obs, los, *, time=None, direction='dep', derivs=False,
                  guess=None, hints=None):
        """The position where a specified line of sight intercepts the Surface.

        Because a NullSurface has no extent, the returned position and parameter values
        are entirely masked.

        Parameters:
            obs (Vector3): Observer position as a Vector3 relative to this Surface's
                origin and frame.
            los (Vector3): Line of sight as a Vector3 in this Surface's frame.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            direction (str, optional): 'arr' for a photon arriving at the surface; 'dep'
                for a photon departing from the surface; ignored.
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
            * `t` (Scalar): Such that `intercept = obs + t * los`.
            * `hints` (Any): The input value of `hints`, included if it is not None.
        """

        # This is a quick way to create a position vector of the correct shape, and with
        # the correct set of derivatives, even though it will be entirely masked.

        pos = (Vector3.as_vector3(obs, recursive=derivs)
               + Vector3.as_vector3(los, recursive=derivs))
        t = pos.to_scalar(0, recursive=derivs)

        pos = pos.as_all_constant().as_all_masked()
        t = t.as_all_constant(0.).as_all_masked()

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
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.
            derivs (bool, optional): True to propagate any derivatives of pos into the
                returned normal vectors.
            hints (Any, optional): Any data that might be useful to carry over from one
                call to the next; unused by this Surface subclass. If it is not None,
                its value is appended to the returned tuple.

        Returns:
            Vector3 or tuple[Vector3, Any]: Directions normal to the Surface that pass
            through the position, optionally followed by `hints`. Vector lengths are
            arbitrary, and the input value of `hints` is returned if it is not None.
        """

        # Always the Z-axis
        if hints is not None:
            return (Vector3.ZAXIS, hints)

        return Vector3.ZAXIS

    def velocity(self, pos, *, obs=None, time=None):
        """The local velocity vector at a point within the surface.

        This can be used to describe the orbital motion of ring particles or local wind
        speeds on a planet.

        Parameters:
            pos (Vector3): Positions at or near the Surface relative to this Surface's
                origin and frame.
            obs (Vector3, optional): Observer position relative to this Surface's origin
                and frame; ignored for this Surface subclass.
            time (Scalar, optional): Time at which to evaluate the Surface; ignored for
                this Surface subclass.

        Returns:
            Vector3: Velocities, in units of km/s.
        """

        # Always zero
        return Vector3.ZERO

##########################################################################################
