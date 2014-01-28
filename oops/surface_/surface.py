################################################################################
# oops/surface_/surface.py: Abstract class Surface
#
# 2/5/12 Modified (MRS) - Minor updates for style
# 3/1/12 MRS: Modified convergence criteria in _solve_photons(), added quick
#   dictionary and config file.
# 3/5/12 MRS: Added class function resolution(), which calculates spatial
#   resolution on a surface based on derivatives with respect to pixel
#   coordinates (u,v).
# 3/23/12 MRS - Introduced event_as_coords() and coords_as_event() as
#   alternatives to coords_from_vector3() and vector3_from_coords(). They help
#   to address the problems with virtual surfaces such as Ansa.
# 3/24/12 MRS - _solve_photon() now properly handles an entirely masked event.
# 5/2/12 MRS - fixed bug in event_as_coords().
# 6/6/12 MRS - added t_guess as a standard argument to intercept(), revised
#   _solve_photon() to take advantage of it; added class constants
#   *_DERIVS_ARE_IMPLEMENTED that can inform the calling program immediately if
#   if derivs are implemented, avoiding the need to catch a NotImplementedError.
# 9/8/13 MRS - added methods photon_to_event_by_coords() and
#   photon_from_event_by_coords(). These have been tested successfully for
#   HST Uranus images using the Ansa surface; however, I did not write unit
#   tests.
################################################################################

import numpy as np

import oops.registry as registry
import oops.constants as constants
from oops.array_ import *
from oops.config import QUICK, SURFACE_PHOTONS, LOGGING
from oops.event import Event
from oops.path_ import *

class Surface(object):
    """Surface is an abstract class describing a 2-D object that moves and
    rotates in space. A surface employs an internal coordinate system, not
    necessarily rectangular, in which two primary coordinates define locations
    on the surface, and an optional third coordinate can define points above or
    below that surface. The shape is always fixed.

    Required attributes:
        origin_id       the ID of the path defining the surface's center.
        frame_id        the ID of the frame in which the surface is defined.
    """

    # Class constants to override where derivs are undefined
    coords_from_vector3_DERIVS_ARE_IMPLEMENTED = True
    vector3_from_coords_DERIVS_ARE_IMPLEMENTED = True
    intercept_DERIVS_ARE_IMPLEMENTED = True
    normal_DERIVS_ARE_IMPLEMENTED = True
    intercept_with_normal_DERIVS_ARE_IMPLEMENTED = True
    intercept_normal_to_DERIVS_ARE_IMPLEMENTED = True

    ########################################
    # Each subclass must override...
    ########################################

    def __init__(self):
        """Constructor for a Surface object.

        Every surface must have the following attributes:
            path_id     the ID of a Path object, which defines the motion of a
                        surface's origin point.
            frame_id    the ID of a Frame object, which defines the orientation
                        of the surface. The surface and its coordinates are
                        fixed within this frame.
        """

        pass

    def coords_from_vector3(self, pos, obs=None, axes=2, derivs=False):
        """Converts from position vectors in the internal frame into the surface
        coordinate system.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            obs         a Vector3 of observer observer positions. Ignored for
                        solid surfaces but needed for virtual surfaces.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate with respect to
                        surface position and observer position are returned as
                        well. Using a tuple, you can indicate whether to return
                        partial derivatives on a coordinate-by-coordinate
                        basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.

                        If derivs is True, then the coordinate has extra
                        attributes "d_dpos" and "d_dobs", which contain the
                        partial derivatives with respect to the surface position
                        and the observer position, represented as a MatrixN
                        objects with item shape [1,3].
        """

        pass

    def vector3_from_coords(self, coords, obs, derivs=False):
        """Returns the position where a point with the given surface coordinates
        would fall in the surface frame, given the location of the observer.

        Input:
            coords      a tuple of two or three Scalars defining the coordinates
            obs         position of the observer in the surface frame. Ignored
                        for solid surfaces but needed for virtual surfaces.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to observer and to the coordinates.

        Return:         a unitless Vector3 of intercept points defined by the
                        coordinates.

                        If derivs is True, then pos is returned with subfields
                        "d_dobs" and "d_dcoords", where the former contains the
                        MatrixN of partial derivatives with respect to obs and
                        the latter is the MatrixN of partial derivatives with
                        respect to the coordinates. The MatrixN item shapes are
                        [3,3].

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.
        """

        pass

    def intercept(self, obs, los, derivs=False, t_guess=False):
        """Returns the position where a specified line of sight intercepts the
        surface.

        Input:
            obs         observer position as a Vector3, with optional units.
            los         line of sight as a Vector3, with optional units.
            derivs      True to include the partial derivatives of the intercept
                        point with respect to obs and los.
            t_guess     initial guess at the t array, optional.

        Return:         a tuple (pos, t) where
            pos         a unitless Vector3 of intercept points on the surface,
                        in km.
            t           a unitless Scalar such that:
                            position = obs + t * los

                        If derivs is True, then pos and t are returned with
                        subfields "d_dobs" and "d_dlos", where the former
                        contains the MatrixN of partial derivatives with respect
                        to obs and the latter is the MatrixN of partial
                        derivatives with respect to los. The MatrixN item shapes
                        are [3,3] for the derivatives of pos, and [1,3] for the
                        derivatives of t.
        """

        pass

    def normal(self, pos, derivs=False):
        """Returns the normal vector at a position at or near a surface.

        Input:
            pos         a Vector3 of positions at or near the surface, with
                        optional units.
            derivs      True to include a matrix of partial derivatives.

        Return:         a unitless Vector3 containing directions normal to the
                        surface that pass through the position. Lengths are
                        arbitrary.

                        If derivs is True, then the normal vectors returned have
                        a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        pass

    ########################################
    # Optional Methods...
    ########################################

    def intercept_with_normal(self, normal, derivs=False):
        """Constructs the intercept point on the surface where the normal vector
        is parallel to the given vector. This is only needed for a limited set
        of subclasses such as Spheroid.

        Input:
            normal      a Vector3 of normal vectors, with optional units.
            derivs      true to return a matrix of partial derivatives.

        Return:         a unitless Vector3 of surface intercept points, in km.
                        Where no solution exists, the components of the returned
                        vector should be masked.

                        If derivs is True, then the returned intercept points
                        have a subfield "d_dperp", which contains the partial
                        derivatives with respect to components of the normal
                        vector, as a MatrixN object with item shape [3,3].
        """

        raise NotImplementedError("intercept_with_normal() not implemented " +
                                  "for class " + type(self).__name__)

    def intercept_normal_to(self, pos, derivs=False):
        """Constructs the intercept point on the surface where a normal vector
        passes through a given position. This is only needed for a limited set
        of subclasses such as Spheroid.

        Input:
            pos         a Vector3 of positions near the surface, with optional
                        units.

        Return:         a unitless vector3 of surface intercept points. Where no
                        solution exists, the returned vector should be masked.

                        If derivs is True, then the returned intercept points
                        have a subfield "d_dpos", which contains the partial
                        derivatives with respect to components of the given
                        position vector, as a MatrixN object with item shape
                        [3,3].
        """

        raise NotImplementedError("intercept_normal_to() not implemented " +
                                  "for class " + type(self).__name__)

    def velocity(self, pos):
        """Returns the local velocity vector at a point within the surface.
        This can be used to describe the orbital motion of ring particles or
        local wind speeds on a planet.

        Input:
            position    a Vector3 of positions at or near the surface, with
                        optional units.

        Return:         a unitless Vector3 of velocities, in units of km/s.
        """

        return Vector3.ZERO

    ############################################################################
    # Generally should not require overrides
    ############################################################################

    # 3/23 MRS - This alternative to as_coords() should be adaptable to virtual
    # surfaces. Tested and in use by Backplane class.
    def event_as_coords(self, event, axes=3, derivs=False):
        """Converts an event object to coordinates and, optionally, their
        derivatives. It uses the linked object to determine the direction to
        the observer.

        Input:
            event       an event occurring at or near the surface.
            subfield    "arr" if this event is defined by arriving photons from
                        another event; "dep" if this event is defined by photons
                        departing to another event.
            axes        2 or 3, indicating whether to return a tuple of two or
                        three Scalar objects.
            derivs      a boolean or tuple of booleans. If True, then the
                        partial derivatives of each coordinate are returned as
                        well. Using a tuple, you can indicate whether to return
                        time derivatives on a coordinate-by-coordinate basis.

        Return:         coordinate values packaged as a tuple containing two or
                        three unitless Scalars, one for each coordinate.

                        Where derivs is True, then each coordinate returned will
                        have a subfield "d_dt" inserted or updated with the
                        rate of change.
        """

        # Locate the event WRT the surface frame if it is not already there
        any_derivs = np.any(derivs)
        wrt_surface = event.wrt(self.origin_id, self.frame_id,
                                derivs=any_derivs)

        # Define the connection to the observer, if any, in the surface frame
        if event.sign == 0:
            obs = None
        elif event.sign < 0:
            obs = event.pos + event.dep_lt * event.dep.unit() * constants.C
        else:
            obs = event.pos + event.arr_lt * event.arr.unit() * constants.C

        # Make sure the position array has the same shape as the event
        pos = wrt_surface.pos.rebroadcast(wrt_surface.shape)

        # Evaluate the coords and optional derivatives
        coords = self.coords_from_vector3(pos, obs=obs, axes=axes,
                                                        derivs=derivs)

        if not any_derivs: return coords

        # Update the derivatives WRT time if necessary
        vel = wrt_surface.vel.rebroadcast(wrt_surface.shape)

        for coord in coords:
            if "d_dpos" in coord.subfields.keys():
                coord.add_to_subfield("d_dt", (coord.d_dpos *
                                               vel.as_column()).as_scalar())

           # This part not yet tested but I think it should work. 3/23 MRS
            if "d_dobs" in coord.subfields.keys() and event.link is not None:
                vlink = event.link.vel
                coord.add_to_subfield("d_dt", (coord.d_dobs *
                                               vlink.as_column()).as_scalar())

        return coords

    # 3/23 MRS currently used in OrbitFrame unit tests
    # May need to be modified for further use
    def coords_as_event(self, time, coords, dcoords_dt=None, obs=None):
        """Converts a time and coordinates in the surface's internal coordinate
        system into an event object.

        Input:
            time        the Scalar of time values at which to evaluate the
                        coordinates.
            coords      a tuple containing the two or three coordinate values.
                        If only two are provided, the third is assumed to be
                        zero.
            dcoords_dt  an optional tuple containing rates of changes of the
                        coordinates. If provided, these values define the
                        velocity vector of the event object.
            obs         a Vector3 of observer positions. In some cases, a
                        surface is defined in part by the position of the
                        observer.

        Note that the coordinates can all have different shapes, but they must
        be broadcastable to a single shape.

        Return:         an event object relative to the origin and frame of the
                        surface.
        """

        if len(coords) == 2:
            (coord1, coord2) = coords
            coord3 = Scalar((0,0,0))
        else:
            (coord1, coord2, coord3) = coords

        derivs = (dcoords_dt is not None)
        pos = self.vector3_from_coords((coord1, coord2, coord3), obs=obs,
                                                                 derivs=derivs)
        vel = self.velocity(pos)

        if derivs:
            if len(dcoords_dt) == 2:
                (dcoord1_dt, dcoord2_dt) = dcoords_dt
                dcoord3_dt = Scalar(0.)
            else:
                (dcoord1_dt, dcoord2_dt, dcoord3_dt) = dcoords_dt

            (dpos_dcoord1,
             dpos_dcoord2, dpos_dcoord3) = pos.d_dcoord.as_columns()

            vel = vel + (dpos_dcoord1 * dcoord1_dt +
                         dpos_dcoord2 * dcoord2_dt +
                         dpos_dcoord3 * dcoord3_dt).as_vector3()
            pos = pos.plain()

        return Event(time, pos, vel, self.origin_id, self.frame_id)

################################################################################
# Photon Solver based on event line of sight
################################################################################

    def photon_from_event(self, link, quick=None, derivs=False,
                                iters=None, precision=None, limit=None):
        """Returns the photon arrival event at the body's surface, for photons
        departing earlier from the specified linking event. Defined by the
        photon departure line of sight at the linking event. See _solve_photon()
        for details."""

        return self._solve_photon(link, +1, quick, derivs,
                                        iters, precision, limit)

    def photon_to_event(self, link, quick=None, derivs=False,
                        iters=None, precision=None, limit=None):
        """Returns the photon departure event at the body's surface, for photons
        arriving later at the specified linking event. Defined by the photon
        arrival line of sight at the linking event. See _solve_photon() for
        details."""

        return self._solve_photon(link, -1, quick, derivs,
                                  iters, precision, limit)

    def _solve_photon(self, link, sign, quick=None, derivs=False,
                      iters=None, precision=None, limit=None):
        """Solve for the event object located on the body's surface that falls
        at the other end of the photon's path to or from a linking event.
        Defined by the photon line of sight at the linking event.

        Input:
            link        the linking event of a photon's arrival or departure.

            sign        -1 to return earlier events, corresponding to photons
                           departing from the surface and arriving later at the
                           linking event. The solution is defined by the vector
                           of arriving photons at the link event.
                        +1 to return later events, corresponding to photons
                           departing from the linking event and arriving later
                           at the surface. The solution is defined by the
                           vector of departing photons at the link event.

            quick       False to disable QuickPaths and QuickFrames, True to use
                        default parameters; a dictionary to override selected
                        default parameters.

            derivs      True to include subfields containing the partial
                        derivatives with respect to the time and line of sight
                        of the linking event.

            The following input parameters have default defined in file
            oops_.config.SURFACE_PHOTONS:

            iters       the maximum number of iterations of Newton's method to
                        perform. It should almost never need to be > 5.
            precision   iteration stops when the largest change in light travel
                        time between one iteration and the next falls below this
                        threshold (in seconds).
            limit       the maximum allowed absolute value of the change in
                        light travel time from the nominal range calculated
                        initially. Changes in light travel with absolute values
                        larger than this limit are clipped. Can be used to
                        limit divergence of the solution in some rare cases.

        Return:         Returns the surface event describing the arrival or
                        departure of the photon. The surface normal and velocity
                        field subfields "perp" and "vflat" are always filled in.
                        The subfields (arr, arr_lt) or (dep, dep_lt) are filled
                        in for arriving or departing photons, respectively.

                        If derivs is True, then these subfields are included:
                        time.d_dlos, pos.d_dlos, time.d_dt, and pos.d_dt. All
                        partial derivative of the surface event are given with
                        respect to the time and line of sight of the linking
                        event.
        """

        # Interpret args
        if iters is None:
            iters = SURFACE_PHOTONS.max_iterations
        if precision is None:
            precision = SURFACE_PHOTONS.dlt_precision
        if limit is None:
            limit = SURFACE_PHOTONS.dlt_limit

        # Interpret the sign
        signed_c = sign * constants.C
        if sign < 0.:
            surface_key = "dep"
            link_key = "arr"
        else:
            link_key = "dep"
            surface_key = "arr" 

        # Define the surface path and frame relative to the SSB in J2000
        origin_wrt_ssb  = registry.connect_paths(self.origin_id, "SSB", "J2000")
        frame_wrt_j2000 = registry.connect_frames(self.frame_id, "J2000")

        # Define the observer and line of sight in the SSB frame
        link_wrt_ssb = link.wrt_ssb(quick)
        obs_wrt_ssb = link_wrt_ssb.pos
        los_wrt_ssb = link_wrt_ssb.subfields[link_key].unit() * constants.C

        # Make an initial guess at the light travel time using the range to the
        # surface's origin
        lt = (obs_wrt_ssb -
              origin_wrt_ssb.event_at_time(link.time).pos).norm() / signed_c
        lt_min = lt.min()
        lt_max = lt.max()

        # If the link is entirely masked...
        if lt_min == lt_max:
          if np.all(link.mask):
            return self._masked_link(link, sign, derivs)

        # Expand the interval
        lt_min -= limit
        lt_max += limit

        # Interpret the quick parameters
        if quick is False:
            quick_dict = False
        else:
            if type(quick) == type({}):
                quickdict = dict(QUICK.dictionary, **quick)
            else:
                quickdict = QUICK.dictionary
            quickdict = dict(quickdict, **{"path_extension": limit,
                                           "frame_extension": limit})

        # Iterate. Convergence is rapid because all speeds are non-relativistic
        max_dlt = np.inf
        new_lt = None
        for iter in range(iters):

            # Evaluate the current time
            surface_time = link.time + lt

            # Quicken when needed
            origin_wrt_ssb = origin_wrt_ssb.quick_path(surface_time,
                                                       quick=quickdict)
            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                          quick=quickdict)

            # Locate the photons relative to the current origin in SSB/J2000
            pos_in_j2000 = (obs_wrt_ssb + lt * los_wrt_ssb
                            - origin_wrt_ssb.event_at_time(surface_time).pos)

            # Rotate into the surface-fixed frame
            surface_xform = frame_wrt_j2000.transform_at_time(surface_time)
            pos_wrt_surface = surface_xform.rotate(pos_in_j2000)
            los_wrt_surface = surface_xform.rotate(los_wrt_ssb)
            obs_wrt_surface = pos_wrt_surface - lt * los_wrt_surface

            # Update the intercept times; save the intercept positions
            (pos_wrt_surface, new_lt) = self.intercept(obs_wrt_surface,
                                                       los_wrt_surface,
                                                       t_guess = new_lt)

            new_lt = new_lt.clip(lt_min, lt_max)
            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max()

            if LOGGING.surface_iterations:
                print LOGGING.prefix, "Surface._solve_photon", iter, max_dlt

            if type(max_dlt) == Scalar and np.all(max_dlt.mask):
                return self._masked_link(link, sign, derivs)

            if max_dlt <= precision or max_dlt >= prev_max_dlt: break

        # Update the mask on light time to hide intercepts behind the observer
        # or outside the defined limit
        lt.mask = (link.mask | lt.mask | (lt.vals * sign < 0.) |
                   (lt.vals == lt_min) | (lt.vals == lt_max))
        if not np.any(lt.mask): lt.mask = False

        # Evaluate the current time
        surface_time = link.time + lt

        # Locate the photons relative to the current origin in SSB/J2000
        pos_in_j2000 = (obs_wrt_ssb + lt * los_wrt_ssb
                            - origin_wrt_ssb.event_at_time(surface_time).pos)

        # Rotate into the surface-fixed frame
        surface_xform = frame_wrt_j2000.transform_at_time(surface_time, quick)
        pos_wrt_surface = surface_xform.rotate(pos_in_j2000)
        los_wrt_surface = surface_xform.rotate(los_wrt_ssb)

        # Create the surface event in its own frame
        surface_event = Event(surface_time, pos_wrt_surface, Vector3.ZERO,
                              self.origin_id, self.frame_id,
                              link = link, sign = sign)
        surface_event.collapse_time()

        # Fill in derivatives if necessary
        if derivs:

            obs_wrt_surface = pos_wrt_surface - lt * los_wrt_surface
            los_wrt_surface = los_wrt_surface.unit()
            (pos, dist) = self.intercept(obs_wrt_surface, los_wrt_surface,
                                                          derivs=True)

            dtime_dobs = dist.d_dobs / signed_c
            dtime_dlos = dist.d_dlos / signed_c
            dpos_dobs = pos.d_dobs
            dpos_dlos = pos.d_dlos

            # Transform derivatives WRT link position to derivatives WRT time
            delta_v = surface_xform.rotate(link_wrt_ssb.vel) - surface_event.vel
            dtime_dt = dist.d_dobs * delta_v / signed_c
            dpos_dt  = pos.d_dobs  * delta_v

            # Set up transform between link frame and surface frame
            j2000_link_frame = registry.connect_frames("J2000", link.frame_id)
            j2000_link_xform = j2000_link_frame.transform_at_time(link.time,
                                                                  quick)
            xform = surface_xform.rotate_transform(j2000_link_xform)

            # Transform derivatives WRT los to los in link frame
            dtime_dlos = xform.unrotate(dtime_dlos.T()).T()
            dpos_dlos  = xform.unrotate(dpos_dlos.T()).T()

            # Save the derivatives as subfields
            surface_event.time.insert_subfield("d_dt",   dtime_dt)
            surface_event.time.insert_subfield("d_dlos", dtime_dlos)
            surface_event.pos.insert_subfield( "d_dt",   dpos_dt)
            surface_event.pos.insert_subfield( "d_dlos", dpos_dlos)

        surface_event.insert_subfield("perp",  self.normal(pos_wrt_surface))
        surface_event.insert_subfield("vflat", self.velocity(pos_wrt_surface))

        surface_event.insert_subfield(surface_key, los_wrt_surface)
        surface_event.insert_subfield(surface_key + "_lt", -lt)

        return surface_event

    def _masked_link(self, link, sign, derivs=False):
        """Returns an entirely masked surface event."""

        surface_event = link.masked_link(self.origin_id, self.frame_id, sign)

        if derivs:
            shape = link.shape
            surface_event.time.insert_subfield("d_dt",
                                        Scalar.all_masked(shape))
            surface_event.time.insert_subfield("d_dlos",
                                        MatrixN.all_masked(shape, item=[1,3]))
            surface_event.pos.insert_subfield( "d_dt",
                                        MatrixN.all_masked(shape, item=[3,1]))
            surface_event.pos.insert_subfield( "d_dlos",
                                        MatrixN.all_masked(shape, item=[3,3]))

        return surface_event

################################################################################
# Photon Solver based on coordinates at the surface
################################################################################

    def photon_from_event_by_coords(self, link, coords,
                                    quick=None, derivs=False, update=True,
                                    iters=None, precision=None, limit=None):
        """Returns the photon arrival event at the body's surface, for photons
        departing earlier from the specified linking event and arriving at the
        specified coordinates. See _solve_photon_by_coords() for details."""

        return self._solve_photon_by_coords(link, coords, +1,
                                            quick, derivs, update,
                                            iters, precision, limit)

    def photon_to_event_by_coords(self, link, coords,
                                  quick=None, derivs=False, update=True,
                                  iters=None, precision=None, limit=None):
        """Returns the photon departure event at the body's surface, for photons
        arriving later at the specified linking event and departing at the
        specified coordinates. See _solve_photon_by_coords() for details."""

        return self._solve_photon_by_coords(link, coords, -1,
                                            quick, derivs, update,
                                            iters, precision, limit)

    def _solve_photon_by_coords(self, link, coords, sign,
                                      quick=None, derivs=False, update=True,
                                      iters=None, precision=None, limit=None):
        """Solve for the event object located on the body's surface that falls
        at the other end of the photon's path to or from a linking event.
        Defined by the coordinates at the surface.

        Input:
            link        the linking event of a photon's arrival or departure.

            coords      a tuple of two or three coordinate values defining the
                        photon's departure or arrival at or near the surface.

            sign        -1 to return earlier events, corresponding to photons
                           departing from the specified surface coordinates and
                           arriving later at the linking event.
                        +1 to return later events, corresponding to photons
                           departing from the linking event and arriving later
                           at the specified surface coordinates.

            quick       False to disable QuickPaths and QuickFrames, True to use
                        default parameters; a dictionary to override selected
                        default parameters.

            derivs      True to include subfields containing the partial
                        derivatives with respect to the time and line of sight
                        of the linking event.

            update      True to update the photon arrival or departure event in
                        the linking event; False to leave the linking event
                        unchanged.

            The following input parameters have default defined in file
            oops_.config.SURFACE_PHOTONS:

            iters       the maximum number of iterations of Newton's method to
                        perform. It should almost never need to be > 5.
            precision   iteration stops when the largest change in light travel
                        time between one iteration and the next falls below this
                        threshold (in seconds).
            limit       the maximum allowed absolute value of the change in
                        light travel time from the nominal range calculated
                        initially. Changes in light travel with absolute values
                        larger than this limit are clipped. Can be used to
                        limit divergence of the solution in some rare cases.

        Return:         Returns the surface event describing the arrival or
                        departure of the photon. The surface normal and velocity
                        field subfields "perp" and "vflat" are always filled in.
                        The subfields (arr, arr_lt) or (dep, dep_lt) are filled
                        in for arriving or departing photons, respectively.

                        If derivs is True, then these subfields are included:
                        time.d_dt, time.d_dpos, pos.d_dt, and pos.d_dpos. These
                        partial derivatives of the surface vent are given with
                        respect to the time and position of the linking event.
        """

        # Interpret args
        if iters is None:
            iters = SURFACE_PHOTONS.max_iterations
        if precision is None:
            precision = SURFACE_PHOTONS.dlt_precision
        if limit is None:
            limit = SURFACE_PHOTONS.dlt_limit

        # Interpret the sign
        signed_c = sign * constants.C
        if sign < 0.:
            surface_key = "dep"
            link_key = "arr"
        else:
            link_key = "dep"
            surface_key = "arr" 

        # Define the surface path and frame relative to the SSB in J2000
        origin_wrt_ssb  = registry.connect_paths(self.origin_id, "SSB", "J2000")
        frame_wrt_j2000 = registry.connect_frames(self.frame_id, "J2000")

        # Define the observer in the SSB frame
        link_wrt_ssb = link.wrt_ssb(quick)
        obs_wrt_ssb = link_wrt_ssb.pos

        # Make an initial guess at the light travel time using the range to the
        # surface's origin
        lt = (obs_wrt_ssb -
              origin_wrt_ssb.event_at_time(link.time).pos).norm() / signed_c
        lt_min = lt.min()
        lt_max = lt.max()

        # If the link is entirely masked...
        if lt_min == lt_max:
          if np.all(link.mask):
            return self._masked_link(link, sign, derivs)

        # Expand the interval
        lt_min -= limit
        lt_max += limit

        # Interpret the quick parameters
        if quick is False:
            quick_dict = False
        else:
            if type(quick) == type({}):
                quickdict = dict(QUICK.dictionary, **quick)
            else:
                quickdict = QUICK.dictionary
            quickdict = dict(quickdict, **{"path_extension": limit,
                                           "frame_extension": limit})

        # Iterate. Convergence is rapid because all speeds are non-relativistic
        max_dlt = np.inf
        new_lt = None

        # For a non-virtual surface, pos_wrt_origin is fixed
        if not self.IS_VIRTUAL:
            pos_wrt_origin = self.vector3_from_coords(coords)

        for iter in range(iters):

            # Evaluate the current time
            surface_time = link.time + lt

            # Quicken the surface path and frame when needed
            origin_wrt_ssb = origin_wrt_ssb.quick_path(surface_time,
                                                       quick=quickdict)
            frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                          quick=quickdict)

            # Evaluate the observer position relative to the current surface
            origin_wrt_ssb_now = origin_wrt_ssb.event_at_time(surface_time).pos
            obs_wrt_origin_j2000 = obs_wrt_ssb - origin_wrt_ssb_now

            surface_xform = frame_wrt_j2000.transform_at_time(surface_time)
            obs_wrt_origin = surface_xform.rotate(obs_wrt_origin_j2000)

            # Locate the coordinate position relative to the current surface,
            # if necessary
            if self.IS_VIRTUAL:
                pos_wrt_origin = self.vector3_from_coords(coords,
                                                          obs_wrt_origin)

            # Update the intercept times
            pos_wrt_origin_j2000 = surface_xform.unrotate(pos_wrt_origin)
            new_lt = (pos_wrt_origin_j2000 + origin_wrt_ssb_now -
                      obs_wrt_ssb).norm() / signed_c

            new_lt = new_lt.clip(lt_min, lt_max)
            dlt = new_lt - lt
            lt = new_lt

            # Test for convergence
            prev_max_dlt = max_dlt
            max_dlt = abs(dlt).max()

            if LOGGING.surface_iterations:
                print LOGGING.prefix, "Surface._solve_photon_by_coords",
                print iter, max_dlt

            if type(max_dlt) == Scalar and np.all(max_dlt.mask):
                return self._masked_link(link, sign, derivs)

            if max_dlt <= precision or max_dlt >= prev_max_dlt: break

        # Update the mask on light time to hide intercepts outside the defined
        # limits
        lt.mask = (link.mask | lt.mask | (lt.vals * sign < 0.) |
                   (lt.vals == lt_min) | (lt.vals == lt_max))
        if not np.any(lt.mask): lt.mask = False

        # Evaluate the current time
        surface_time = link.time + lt

        # Quicken the surface path and frame when needed
        origin_wrt_ssb = origin_wrt_ssb.quick_path(surface_time,
                                                   quick=quickdict)
        frame_wrt_j2000 = frame_wrt_j2000.quick_frame(surface_time,
                                                      quick=quickdict)

        # Evaluate the observer position relative to the current surface
        origin_wrt_ssb_now = origin_wrt_ssb.event_at_time(surface_time).pos
        obs_wrt_origin_j2000 = obs_wrt_ssb - origin_wrt_ssb_now

        surface_xform = frame_wrt_j2000.transform_at_time(surface_time)
        obs_wrt_origin = surface_xform.rotate(obs_wrt_origin_j2000)

        # Locate the coordinate position relative to the current surface,
        # if necessary
        if self.IS_VIRTUAL:
            pos_wrt_origin = self.vector3_from_coords(coords, obs_wrt_origin)

        # Determine the line of sight vector in J2000
        pos_wrt_origin_j2000 = surface_xform.unrotate(pos_wrt_origin)
        los_in_j2000 = sign * (pos_wrt_origin_j2000 + origin_wrt_ssb_now -
                               obs_wrt_ssb)

        # Create the surface event in its own frame
        surface_event = Event(surface_time, pos_wrt_origin, Vector3.ZERO,
                              self.origin_id, self.frame_id,
                              link = link, sign = sign)
        surface_event.collapse_time()

        surface_event.insert_subfield("perp",  self.normal(pos_wrt_origin))
        surface_event.insert_subfield("vflat", self.velocity(pos_wrt_origin))

        los_wrt_surface = surface_xform.rotate(los_in_j2000)
        surface_event.insert_subfield(surface_key, los_wrt_surface)
        surface_event.insert_subfield(surface_key + "_lt", -lt)

        # Fill in derivatives if necessary
        if derivs:
            raise NotImplementedException("Derivatives are not implemented " +
                                          "for _solve_photon_by_coords()")

        # Update the linking event if necessary
        if update:
            link_frame = registry.connect_frames(link.frame_id, "J2000")
            link_xform = link_frame.transform_at_time(link.time, quick)

            los_wrt_link = link_xform.rotate(los_in_j2000)

            link.insert_subfield(link_key, los_wrt_link)
            link.insert_subfield(link_key + "_lt", lt)

            # The event is already determined wrt SSB, so just save it
            link_wrt_ssb.insert_subfield(link_key, los_in_j2000)
            link_wrt_ssb.insert_subfield(link_key + "_lt", lt)
            link.filled_ssb = link_wrt_ssb

        return surface_event

################################################################################
# Class Method
################################################################################

    @staticmethod
    def resolution(dpos_duv):
        """Determines the spatial resolution on a surface.

        Input:
            dpos_duv    A MatrixN with item shape [3,2], defining the partial
                        derivatives d(x,y,z)/d(u,v), where (x,y,z) are the 3-D
                        coordinates of a point on the surface, and (u,v) are
                        pixel coordinates.

        Return:         A tuple (res_min, res_max) where:
            res_min     A Scalar containing resolution values (km/pixel) in the
                        direction of finest spatial resolution.
            res_max     A Scalar containing resolution values (km/pixel) in the
                        direction of coarsest spatial resolution.
        """

        # Define vectors parallel to the surface, containing the derivatives
        # with respect to each pixel coordinate.
        dpos_du = Vector3(dpos_duv.vals[...,0], dpos_duv.mask)
        dpos_dv = Vector3(dpos_duv.vals[...,1], dpos_duv.mask)

        # The resolution should be independent of the rotation angle of the
        # grid. We therefore need to solve for an angle theta such that
        #   dpos_du' = cos(theta) dpos_du - sin(theta) dpos_dv
        #   dpos_dv' = sin(theta) dpos_du + cos(theta) dpos_dv
        # where
        #   dpos_du' <dot> dpos_dv' = 0
        #
        # Then, the magnitudes of dpos_du' and dpos_dv' will be the local values
        # of finest and coarsest spatial resolution.
        #
        # Laying aside the overall scaling for a moment, instead solve:
        #   dpos_du' =   dpos_du - t dpos_dv
        #   dpos_dv' = t dpos_du +   dpos_dv
        # for which the dot product is zero.
        #
        # 0 =   t^2 (dpos_du <dot> dpos_dv)
        #     + t   (|dpos_dv|^2 - |dpos_du|^2)
        #     -     (dpos_du <dot> dpos_dv)
        #
        # Use the quadratic formula.

        a = dpos_du.dot(dpos_dv)
        b = dpos_dv.dot(dpos_dv) - dpos_du.dot(dpos_du)
        # c = -a

        # discr = b**2 - 4*a*c
        discr = b**2 + 4*a**2
        t = (discr.sqrt() - b)/ (2*a)

        # Now normalize and construct the primed partials
        cos_theta = 1. / (1 + t**2).sqrt()
        sin_theta = t * cos_theta

        dpos_du_prime_norm = (cos_theta * dpos_du - sin_theta * dpos_dv).norm()
        dpos_dv_prime_norm = (sin_theta * dpos_du + cos_theta * dpos_dv).norm()

        minres = Scalar(np.minimum(dpos_du_prime_norm.vals,
                                   dpos_dv_prime_norm.vals), dpos_duv.mask)
        maxres = Scalar(np.maximum(dpos_du_prime_norm.vals,
                                   dpos_dv_prime_norm.vals), dpos_duv.mask)

        return (minres, maxres)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Surface(unittest.TestCase):

    def runTest(self):

        # TBD

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
