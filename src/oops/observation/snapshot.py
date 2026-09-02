##########################################################################################
# oops/observation/snapshot.py
##########################################################################################

import numpy as np

from polymath                 import Scalar, Pair, Vector, Vector3, Qube
from oops.observation         import Observation
from oops.body                import Body
from oops.cadence             import Cadence
from oops.cadence.snapcadence import SnapCadence
from oops.event               import Event
from oops.frame               import Frame
from oops.path                import Path
from oops.path.multipath      import MultiPath


class Snapshot(Observation):
    """A Snapshot is an Observation consisting of a 2-D image made up of pixels all
    exposed at the same time.
    """

    _INVENTORY_IMPLEMENTED = True

    def __init__(self, axes, tstart, texp, fov, path, frame, **subfields):
        """Constructor for a Snapshot.

        Parameters:
            axes (list or tuple): Strings, with one value for each axis in the associated
                data array. A value of 'u' should appear at the location of the array's
                u-axis; 'v' should appear at the location of the array's v-axis. For
                example, ('v','u'), is correct for a 2-D array read from an image file in
                FITS or VICAR format.
            tstart (float): The start time of the observation in seconds TDB.
                Alternatively, a Cadence object with shape (1,) defining `tstart` and
                `texp`.
            texp (float): Exposure duration of the observation in seconds. Ignored if
                `tstart` is specified as a Cadence.
            fov (FOV): (field-of-view) object, which describes the field of view including
                any spatial distortion. It maps between spatial coordinates (u,v) and
                instrument coordinates (x,y).
            path (Path): The path waypoint co-located with the instrument.
            frame (Frame): The wayframe of a coordinate frame fixed to the optics of the
                instrument. This frame should have its Z-axis pointing outward near the
                center of the line of sight, with the X-axis pointing rightward and the
                Y-axis pointing downward.
            subfields (dict): All of the optional attributes. Additional subfields may be
                included as needed.

        Raises:
            ValueError: If `axes` does not contain both 'u' and 'v', or if `tstart` is a
                Cadence whose shape is not (1,).
        """

        # Basic properties
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        # FOV
        self.fov = fov
        self.uv_shape = tuple(self.fov.uv_shape.vals)

        # Axes
        self._axes = list(axes)
        self.u_axis = self._axes.index('u')
        self.v_axis = self._axes.index('v')
        self.swap_uv = (self.u_axis > self.v_axis)

        self.t_axis = -1

        # Shape / Size
        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self.uv_shape[0]
        self.shape[self.v_axis] = self.uv_shape[1]

        # Cadence
        if isinstance(tstart, Cadence):
            self.cadence = tstart
            if self.cadence.shape != (1,):
                raise ValueError('Shape of Snapshot cadence must be (1,)')
            self._texp = self.cadence.time[1] - self.cadence.time[0]
        else:
            self.cadence = SnapCadence(tstart, texp)
            self._texp = texp

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

    def __getstate__(self):
        self.refresh()
        return (self._axes, self.cadence, self._texp, self.fov, self.path, self.frame,
                self.subfields)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])
        self.freeze()

    def uvt(self, indices, *, remask=False, derivs=True):
        """Coordinates `(u,v)` and time `t` for indices into the data array.

        This method supports non-integer index values.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.
            derivs (bool, optional): True to include derivatives in the returned values.

        Returns:
            tuple[Pair, Scalar]: `(uv, time)`, where `uv` defines the values of `(u,v)`
            within the FOV that are associated with the array indices and `time` defines
            the time in seconds TDB associated with the array indices.
        """

        indices = Vector.as_vector(indices, recursive=derivs)
        uv = indices.to_pair((self.u_axis, self.v_axis))
        time = Scalar(self.cadence.midtime)

        if remask:
            is_outside = self.uv_is_outside(uv, inclusive=True)
            if np.any(is_outside.vals):
                uv = uv.remask_or(is_outside.vals)
                time = Scalar.filled(uv.shape, self.midtime, mask=uv.mask)

        return (uv, time)

    def uvt_range(self, indices, *, remask=False):
        """Ranges of `(u,v)` spatial coordinates and time for integer array indices.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Pair, Pair, Scalar, Scalar]: `(uv_min, uv_max, time_min, time_max)`,
            where:

            * `uv_min`: The minimum values of (u,v) associated with the pixel.
            * `uv_max`: The maximum values of (u,v).
            * `time_min`: The minimum time associated with the pixel, in seconds TDB.
            * `time_max`: The maximum time value.
        """

        indices = Vector.as_vector(indices, recursive=False)
        uv = indices.to_pair((self.u_axis, self.v_axis))
        uv_min = uv.int(top=self.uv_shape, remask=remask)

        # Times can be returned "shapeless" unless a mask is needed
        time_min = Scalar.filled(uv.shape, self.time[0], mask=uv_min.mask)
        time_max = Scalar.filled(uv.shape, self.time[1], mask=uv_min.mask)

        return (uv_min, uv_min + Pair.INT11, time_min, time_max)

    def uv_range_at_tstep(self, tstep, *, remask=False):
        """The range of spatial `(u,v)` pixels active at a particular time step.

        Every pixel of a Snapshot is exposed at the same time, so the range always covers
        the whole FOV.

        Parameters:
            tstep (Scalar): Time step index.
            remask (bool, optional): True to mask values outside the time interval.

        Returns:
            tuple[Pair, Pair]: `(uv_min, uv_max)`, where `uv_min` is the minimum value of
            the FOV `(u,v)` coordinates active at this time step and `uv_max` is the
            maximum value, exclusive.
        """

        return Observation.uv_range_at_tstep_0d(self, tstep, uv_shape=self.uv_shape,
                                                remask=remask)

    def time_range_at_uv(self, uv_pair, *, remask=False):
        """The start and stop times of the specified spatial pixel `(u,v)`.

        Every pixel of a Snapshot is exposed at the same time, so these are the start and
        stop times of the observation overall.

        Parameters:
            uv_pair (Pair): Spatial (u,v) data array coordinates, truncated to integers if
                necessary.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: Scalars of the start time and stop time of each `(u,v)`
            pair, as seconds TDB.
        """

        uv_pair = Pair.as_pair(uv_pair)

        if remask or np.any(uv_pair.mask):
            is_outside = self.uv_is_outside(uv_pair, inclusive=True)
            new_mask = Qube.or_(is_outside.vals, uv_pair.mask)
            if new_mask is not False:
                time_min = Scalar.filled(uv_pair.shape, self.time[0], mask=new_mask)
                time_max = Scalar.filled(uv_pair.shape, self.time[1], mask=new_mask)
                return (time_min, time_max)

        # Without a mask, it's OK to return shapeless values
        return (Scalar(self.cadence.time[0]), Scalar(self.cadence.time[1]))

    def uv_range_at_time(self, time, *, remask=False):
        """The `(u,v)` range of spatial pixels in the data array observed at the specified
        time.

        Every pixel of a Snapshot is exposed at the same time, so this range always covers
        the whole FOV.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.

        Returns:
            tuple[Pair, Pair]: `(uv_min, uv_max)`, where `uv_min` is the lower corner of
            the `(u,v)` rectangle observed and `uv_max` is the upper corner.
        """

        return Observation.uv_range_at_time_0d(self, time, uv_shape=self.fov.uv_shape,
                                               remask=remask)

    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Parameters:
            dtime (float): The time offset to apply to the observation, in units of
                seconds. A positive value shifts the observation later.

        Returns:
            Observation: A (shallow) copy of the object with a new time.
        """

        cadence = self.cadence.time_shift(dtime)
        return Snapshot(axes=self._axes, tstart=cadence, texp=self._texp, fov=self.fov,
                        path=self.path, frame=self.frame, **self.subfields)

    ######################################################################################
    # Overrides of Observation methods
    ######################################################################################

    def uv_from_ra_and_dec(self, ra, dec, *, tfrac=0.5, time=None, apparent=True,
                           derivs=False, iters=2, quick=None):
        """Convert arbitrary scalars of RA and dec to FOV `(u,v)` coordinates.

        Parameters:
            ra (Scalar): J2000 right ascensions.
            dec (Scalar): J2000 declinations.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where tfrac=0 at the beginning and 1 at the end. Default is 0.5. Ignored
                if `time` is specified.
            time (Scalar, optional): Scalar of optional absolute time in seconds.
            apparent (bool, optional): True to interpret the (RA,dec) values as apparent
                coordinates; False to interpret them as actual coordinates. Default is
                True.
            derivs (bool, optional): True to propagate derivatives of ra and dec through
                to derivatives of the returned (u,v) Pairs.
            iters (int, optional): Ignored; a Snapshot always converges in one iteration.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.

        Returns:
            Pair: `(u,v)` coordinates.

        Notes:
            The only reasons for iteration are that the C-matrix and the velocity WRT the
            SSB could vary during the observation. This can be neglected for a Snapshot.
        """

        # Limit iterations to 1 for Snapshot
        return super(Snapshot, self).uv_from_ra_and_dec(ra, dec, tfrac=tfrac, time=time,
                                                        apparent=apparent, derivs=derivs,
                                                        iters=1, quick=quick)

    def uv_from_path(self, path, *, tfrac=0.5, time=None, derivs=False, guess=None,
                     quick=None, converge=None):
        """The `(u,v)` indices of an object in the FOV, given its path.

        Because every pixel of a Snapshot is exposed at the same time, `tfrac` is
        converted to an absolute time and no iteration is required.

        Parameters:
            path (Path): Object.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where tfrac=0 at the beginning and 1 at the end. Default is 0.5.
            time (Scalar, optional): Scalar of optional absolute time in seconds. If
                specified, `tfrac` is ignored.
            derivs (bool, optional): True to propagate derivatives of the link time and
                position into the returned event.
            guess (Scalar, optional): An optional guess at the light travel time from the
                path to the event.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            converge (dict, optional): Parameters to override the configured default
                convergence parameters. The default configuration is defined in config.py.

        Returns:
            Pair: The `(u,v)` indices of the pixel in which the point was found.
        """

        # Convert tfrac to time. That way, iteration is avoided
        if time is None:
            time = self.time[0] + tfrac * (self.time[1] - self.time[0])

        return super(Snapshot, self).uv_from_path(path, tfrac=tfrac, time=time,
                                                  derivs=derivs, guess=guess,
                                                  quick=quick, converge=converge)

    def uv_from_coords(self, surface, coords, *, tfrac=0.5, time=None, underside=False,
                       derivs=False, quick=None, converge=None):
        """The `(u,v)` indices of a surface point, given its coordinates. **** NOT WELL
        TESTED! ****

        Parameters:
            surface (Surface): The Surface object.
            coords (tuple[Scalar, ...]): Two or three surface coordinates. The Scalars
                need not be the same shape, but must broadcast to the same shape.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where tfrac=0 at the beginning and 1 at the end. Default is 0.5.
            time (Scalar, optional): Scalar of optional absolute time in seconds. If
                specified, `tfrac` is ignored.
            underside (bool, optional): True for the underside of the surface (emission >
                90 degrees) to be unmasked.
            derivs (bool, optional): True to propagate derivatives of the link time and
                position into the returned event.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            converge (dict, optional): Parameters to override the configured default
                convergence parameters. The default configuration is defined in config.py.

        Returns:
            Pair: The `(u,v)` indices of the pixel in which the point was found.
        """

        if time is None:
            time = self.time[0] + tfrac * (self.time[1] - self.time[0])

        obs_event = Event(time, Vector3.ZERO, self.path, self.frame)
        (surface_event,
         obs_event) = surface.photon_to_coords(obs_event, coords, derivs=derivs,
                                               quick=quick, converge=converge)

        neg_arr_ap = obs_event.neg_arr_ap
        if not underside:
            normal = surface.normal(surface_event.pos)
            mask = (normal.dot(surface_event.dep_ap, recursive=False) < 0.)
            neg_arr_ap = neg_arr_ap.remask_or(mask)

        return self.fov.uv_from_los_t(neg_arr_ap, time=time, derivs=derivs)

    def inventory(self, bodies, *, tfrac=0.5, time=None, expand=0., cache=True,
                  return_type='list', fov=None, quick=None, converge=None):
        """Info about the bodies that appear unobscured inside the FOV.

        Restrictions: All inventory calculations are performed at a single observation
        time specified by tfrac. All bodies are assumed to be spherical.

        Parameters:
            bodies (list): The names of the body objects to be included in the inventory.
            tfrac (Scalar, optional): Fractional time from the beginning to the end of the
                observation for which the inventory applies. 0 for the beginning; 0.5 for
                the midtime, 1 for the end time. Ignored if time is specified.
            time (Scalar, optional): Scalar of optional absolute time in seconds.
            expand (float, optional): An optional angle in radians by which to extend the
                limits of the field of view. This can be used to accommodate pointing
                uncertainties.
            cache (bool, optional): If False, do not cache the body paths. Default is
                True.
            return_type (str, optional): One of "list", "flags", or "full":

                * "list": Return the inventory as a list of names.
                * "flags": Return the inventory as an array of boolean flag values in the
                  same order as bodies.
                * "full": Return the inventory as a dictionary of dictionaries. The main
                  dictionary is indexed by body name. The subdictionaries contain
                  attributes of the body in the FOV.

            fov (FOV, optional): Use this fov; if None, use `self.fov`.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            converge (dict, optional): Parameters to override the configured default
                convergence parameters. The default configuration is defined in config.py.

        Returns:
            (list, array, or dict).

            * If return_type is "list", it returns a list of the names of all the body
              objects that fall at least partially inside the FOV and are not completely
              obscured by another object in the list.

            * If return_type is "flags", it returns a boolean array containing True
              everywhere that the body falls at least partially inside the FOV and is not
              completely obscured.

            * If return_type is "full", it returns a dictionary with one entry per body,
              whether or not that body falls inside the FOV. Each dictionary entry is
              itself a dictionary containing data about the body in the FOV:

                - "name" (str): The body name.
                - "inside" (bool): True if the body is unobscured inside the FOV.
                - "center_uv" (ndarray): The `(u,v)` coordinates of the center
                  point, as two floats.
                - "center" (ndarray): The direction of the center point, as three
                  floats.
                - "range" (float): The distance in km.
                - "outer_radius" (float): The outer radius of the body in km.
                - "inner_radius" (float): The inner radius of the body in km.
                - "resolution" (ndarray): The resolution in the `(u,v)` directions
                  at the given range, as two floats.
                - "u_min" (int): The minimum `u` value covered by the body, clipped to
                  the FOV boundaries.
                - "u_max" (int): The maximum `u` value covered by the body, clipped to
                  the FOV boundaries.
                - "v_min" (int): The minimum `v` value covered by the body, clipped to
                  the FOV boundaries.
                - "v_max" (int): The maximum `v` value covered by the body, clipped to
                  the FOV boundaries.
                - "u_min_unclipped" (int): Same as "u_min", but not clipped.
                - "u_max_unclipped" (int): Same as "u_max", but not clipped.
                - "v_min_unclipped" (int): Same as "v_min", but not clipped.
                - "v_max_unclipped" (int): Same as "v_max", but not clipped.
                - "u_pixel_size" (float): The diameter of the body in pixels in units of
                  the `u` pixels.
                - "v_pixel_size" (float): The diameter of the body in pixels in units of
                  the `v` pixels.

        Raises:
            ValueError: If `return_type` is not one of "list", "flags", or "full".
        """

        if return_type not in ('list', 'flags', 'full'):
            raise ValueError('invalid return_type for Observation.inventory: '
                             + repr(return_type))

        if fov is None:
            fov = self.fov

        body_names = [Body.as_body_name(body) for body in bodies]
        bodies  = [Body.as_body(body) for body in bodies]
        nbodies = len(bodies)

        path_ids = [body.path for body in bodies]
        path_id = '+' if cache else None
        multipath = MultiPath(path_ids, path_id=path_id)

        if time is None:
            tfrac = Scalar.as_scalar(tfrac)
            obs_time = self.time[0] + tfrac * (self.time[1] - self.time[0])
        else:
            obs_time = Scalar.as_scalar(time)

        obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)
        (_,
         arrival_event) = multipath.photon_to_event(obs_event, quick=quick,
                                                    converge=converge)

        centers = arrival_event.neg_arr_ap
        ranges = centers.norm()
        radii = Scalar([body.radius for body in bodies])
        radius_angles = (radii / ranges).arcsin()

        inner_radii = Scalar([body.inner_radius for body in bodies])
        inner_angles = (inner_radii / ranges).arcsin()

        # This array equals True for each body falling somewhere inside the FOV
        falls_inside = np.empty(nbodies, dtype='bool')
        for i in range(nbodies):
            falls_inside[i] = fov.sphere_falls_inside(centers[i], radii[i], time=obs_time,
                                                      border=expand)

        # This array equals True for each body completely hidden by another
        is_hidden = np.zeros(nbodies, dtype='bool')
        for i in range(nbodies):
            if not falls_inside[i]:
                continue

            for j in range(nbodies):
                if not falls_inside[j]:
                    continue

                if ranges[i] < ranges[j]:
                    continue
                if radius_angles[i] > inner_angles[j]:
                    continue

                sep = centers[i].sep(centers[j])
                if sep < inner_angles[j] - radius_angles[i]:
                    is_hidden[i] = True

        flags = falls_inside & ~is_hidden

        # Return as flags
        if return_type == 'flags':
            return flags

        # Return as list
        if return_type == 'list':
            ret_list = []
            for i in range(nbodies):
                if flags[i]:
                    ret_list.append(body_names[i])
            return ret_list

        # Return full info
        returned_dict = {}

        u_scale = fov.uv_scale.vals[0]
        v_scale = np.abs(fov.uv_scale.vals[1])
        body_uv = fov.uv_from_los_t(arrival_event.neg_arr_ap, time=obs_time).vals
        for i in range(nbodies):
            body_data = {}
            body_data['name'] = body_names[i]
            body_data['inside'] = flags[i]
            body_data['center_uv'] = body_uv[i]
            body_data['center'] = centers[i].vals
            body_data['range'] = ranges[i].vals
            body_data['outer_radius'] = radii[i].vals
            body_data['inner_radius'] = inner_radii[i].vals

            u_res = ranges[i] * fov.uv_scale.to_scalar(0).tan()
            v_res = ranges[i] * fov.uv_scale.to_scalar(1).tan()
            body_data['resolution'] = Pair.from_scalars(u_res, v_res).vals

            u = body_uv[i][0]
            v = body_uv[i][1]
            u_min_unclipped = int(np.floor(u-radius_angles[i].vals/u_scale))
            u_max_unclipped = int(np.ceil( u+radius_angles[i].vals/u_scale))
            v_min_unclipped = int(np.floor(v-radius_angles[i].vals/v_scale))
            v_max_unclipped = int(np.ceil( v+radius_angles[i].vals/v_scale))

            body_data['u_min_unclipped'] = u_min_unclipped
            body_data['u_max_unclipped'] = u_max_unclipped
            body_data['v_min_unclipped'] = v_min_unclipped
            body_data['v_max_unclipped'] = v_max_unclipped

            body_data['u_min'] = np.clip(u_min_unclipped, 0, self.uv_shape[0]-1)
            body_data['u_max'] = np.clip(u_max_unclipped, 0, self.uv_shape[0]-1)
            body_data['v_min'] = np.clip(v_min_unclipped, 0, self.uv_shape[1]-1)
            body_data['v_max'] = np.clip(v_max_unclipped, 0, self.uv_shape[1]-1)

            body_data['u_pixel_size'] = radius_angles[i].vals/u_scale*2
            body_data['v_pixel_size'] = radius_angles[i].vals/v_scale*2

            returned_dict[body_names[i]] = body_data

        return returned_dict

##########################################################################################
