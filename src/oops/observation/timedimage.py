##########################################################################################
# oops/observation/timedimage.py
##########################################################################################

import numpy as np

from polymath                  import Pair, Vector, Qube
from oops.observation          import Observation
from oops.observation.snapshot import Snapshot
from oops.frame                import Frame
from oops.path                 import Path


class TimedImage(Observation):
    """An image in which the individual pixels have distinct timing.

    Pixel timing is defined by a 1-D or 2-D cadence.
    """

    # NOTE:
    # This class now encompasses the earlier subclasses Pushbroom, Pushframe, RasterScan,
    # RasterSlit, and Slit. (In other words, every observation subclass with two spatial
    # dimensions except Snapshot.)

    INVENTORY_IMPLEMENTED = True

    def __init__(self, axes, cadence, fov, path, frame, **subfields):
        """Constructor for a TimedImage.

        Parameters:
            axes (list or tuple): Strings, with one value for each axis in the associated
                data array. One of these strings must begin with "u", and the other must
                begin with "v", to indicate the locations of the spatial axes. If the
                image has a 1-D cadence, then "t" should be appended to the name of the
                axis containing time dependence. If both axes have time dependence, one
                should have the suffix "fast" and the other should have suffix "slow".
            cadence (Cadence): A 1-D or 2-D Cadence object defining the start and stop
                time of each pixel.
            fov (FOV): (field-of-view) object, which describes the field of view including
                any spatial distortion and possible time-dependence. It maps between
                spatial coordinates (u,v) and instrument coordinates (x,y).
            path (Path): The path waypoint co-located with the instrument.
            frame (Frame): The wayframe of a coordinate frame fixed to the optics of the
                instrument. This frame should have its Z-axis pointing outward near the
                center of the line of sight, with the X-axis pointing rightward and the
                Y-axis pointing downward.
            subfields (dict): All of the optional attributes. Additional subfields may be
                included as needed. The subfield `shape` overrides the shape derived from
                the FOV and the cadence.
        """

        # Basic properties
        self.path = Path.as_waypoint(path)
        self.frame = Frame.as_wayframe(frame)

        # Static FOV
        self.fov = fov
        self._fov_shape = tuple(self.fov.uv_shape.vals)
        self._has_unit_fov_axis = self._fov_shape[0] == 1 or self._fov_shape[1] == 1

        # Axes
        self._axes = tuple(axes)

        u_axes = [k for k in range(len(self._axes)) if self._axes[k].startswith('u')]
        v_axes = [k for k in range(len(self._axes)) if self._axes[k].startswith('v')]
        if len(u_axes) != 1 or len(v_axes) != 1:
            raise ValueError(f'invalid axis labels for TimedImage: {self._axes}')

        self.u_axis = u_axes[0]
        self.v_axis = v_axes[0]

        u_suffix = self._axes[self.u_axis][1:]
        v_suffix = self._axes[self.v_axis][1:]

        if u_suffix == 't' and v_suffix == '':
            self.t_axis = self.u_axis
            self._t_uv_axis = 0
        elif u_suffix == '' and v_suffix == 't':
            self.t_axis = self.v_axis
            self._t_uv_axis = 1
        elif u_suffix == 'fast' and v_suffix == 'slow':
            self.t_axis = (self.v_axis, self.u_axis)
            self._fast_t_uv_axis = 0
        elif u_suffix == 'slow' and v_suffix == 'fast':
            self.t_axis = (self.u_axis, self.v_axis)
            self._fast_t_uv_axis = 1
        else:
            raise ValueError('invalid axis labels for TimedImage: '
                             f'"{self._axes[self.u_axis]}", '
                             f'"{self._axes[self.v_axis]}"')

        self.swap_uv = (self.u_axis > self.v_axis)
        self._time_is_1d = not isinstance(self.t_axis, tuple)

        # Cadence
        self.cadence = cadence
        if self._time_is_1d:
            if len(self.cadence.shape) != 1:
                raise ValueError('TimedImage axes requires 1-D cadence')
        else:
            if len(self.cadence.shape) != 2:
                raise ValueError('TimedImage axes requires 2-D cadence')

        # Shape / Size
        self.shape = len(axes) * [0]
        self.shape[self.u_axis] = self._fov_shape[0]
        self.shape[self.v_axis] = self._fov_shape[1]

        # This is the (u,v) shape of the observation, not necessarily that of the FOV
        self.uv_shape = [self._fov_shape[0], self._fov_shape[1]]

        # Cadence overrides the shape as defined by the FOV
        # However, the inventory method will require serious modification for observations
        # in which the cadence defines one dimension, not the FOV.
        if self._time_is_1d:
            t_size = self.cadence.shape[0]
            if t_size < self.shape[self.t_axis]:
                raise ValueError('TimedImage FOV and cadence have incompatible shapes')
            self._extended_fov = (t_size > self.shape[self.t_axis])
            self.shape[self.t_axis] = t_size
            self.uv_shape[self._t_uv_axis] = t_size
        else:
            if self.shape[self.t_axis[0]] not in (self.cadence.shape[0], 1):
                raise ValueError('TimedImage FOV and cadence have incompatible shapes')
            t_size = self.cadence.shape[1]
            if t_size < self.shape[self.t_axis[1]]:
                raise ValueError('TimedImage FOV and cadence have incompatible shapes')
            self._extended_fov = (t_size > self.shape[self.t_axis[1]])
            self.shape[self.t_axis[1]] = t_size
            self.uv_shape[self._fast_t_uv_axis] = t_size

        # Let the user override the shape (to replace zeros if desired)
        self.shape = tuple(self.shape)
        if 'shape' in subfields:
            self.shape = tuple(subfields['shape'])
            del subfields['shape']

        self.INVENTORY_IMPLEMENTED = not self._extended_fov

        # Optional subfields
        self.subfields = {}
        for key in subfields.keys():
            self.insert_subfield(key, subfields[key])

        # TODO: implement inventory and related methods for an extended FOV.

        if self._extended_fov:
            self._snapshot = None
        else:
            snapshot_axes = list(self._axes)    # a copy
            snapshot_axes[self.u_axis] = 'u'
            snapshot_axes[self.v_axis] = 'v'
            snapshot_tstart = self.cadence.time[0]
            snapshot_texp = self.cadence.time[1] - self.cadence.time[0]

            if 'texp' in subfields:             # this creates a conflict
                subfields = subfields.copy()
                subfields['texp_'] = subfields['texp']
                del subfields['texp']

            self._snapshot = Snapshot(snapshot_axes, snapshot_tstart, snapshot_texp,
                                      fov=self.fov, path=self.path, frame=self.frame,
                                      **subfields)

    def __getstate__(self):
        self.refresh()
        return (self._axes, self.cadence, self.fov, Path.as_primary_path(self.path),
                Frame.as_primary_frame(self.frame), self.subfields)

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

        # If an FOV axis has unit length, we always land at 0.5
        if self._has_unit_fov_axis:
            if uv.is_float():
                uv = uv.copy()
            else:
                uv = uv.as_float()

            if self._fov_shape[0] == 1:
                uv.vals[..., 0] = 0.5
            if self._fov_shape[1] == 1:
                uv.vals[..., 1] = 0.5

        new_mask = False
        if self._time_is_1d:
            tstep = indices.to_scalar(self.t_axis)

            # Re-mask the time-independent axis if necessary
            if remask:
                not_t_vals = uv.vals[..., 1 - self._t_uv_axis]
                not_t_max = self.uv_shape[1 - self._t_uv_axis]
                new_mask = (not_t_vals < 0) | (not_t_vals > not_t_max)
                if not np.any(new_mask):
                    new_mask = False
        else:
            tstep = indices.to_pair(self.t_axis)

        time = self.cadence.time_at_tstep(tstep, remask=remask, derivs=derivs)

        # Merge masks if necessary
        if remask:
            new_mask = Qube.or_(new_mask, time.mask)
            uv = uv.remask(new_mask)
            time = time.remask(new_mask)

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

        # If an FOV axis has unit length, we always land at range (0,1)
        if self._has_unit_fov_axis:
            uv_min = uv_min.copy()
            if self._fov_shape[0] == 1:
                uv_min.vals[..., 0] = 0
            if self._fov_shape[1] == 1:
                uv_min.vals[..., 1] = 0

        new_mask = False
        if self._time_is_1d:
            tstep = indices.to_scalar(self.t_axis)

            # Re-mask the time-independent axis if necessary
            if remask:
                not_t_vals = uv.vals[..., 1 - self._t_uv_axis]
                not_t_max = self.uv_shape[1 - self._t_uv_axis]
                new_mask = (not_t_vals < 0) | (not_t_vals > not_t_max)
        else:
            tstep = indices.to_pair(self.t_axis)

        (time_min, time_max) = self.cadence.time_range_at_tstep(tstep, remask=remask)

        # Merge masks if necessary
        if remask:
            if np.any(new_mask):
                new_mask = Qube.or_(new_mask, time_min.mask)
                uv_min = uv_min.remask(new_mask)
                time_min = time_min.remask(new_mask)
                time_max = time_max.remask(new_mask)
            else:
                uv_min = uv_min.remask(time_min.mask)

        return (uv_min, uv_min + Pair.INT11, time_min, time_max)

    def time_range_at_uv(self, uv_pair, *, remask=False):
        """The start and stop times of the specified spatial pixel `(u,v)`.

        The times are those of the cadence, evaluated at the `(u,v)` index or indices that
        carry the time-dependence.

        Parameters:
            uv_pair (Pair): Spatial (u,v) data array coordinates, truncated to integers if
                necessary.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: Scalars of the start time and stop time of each `(u,v)`
            pair, as seconds TDB.
        """

        if self._time_is_1d:
            return self.time_range_at_uv_1d(uv_pair, axis=self._t_uv_axis, remask=remask)
        else:
            return self.time_range_at_uv_2d(uv_pair, fast=self._fast_t_uv_axis,
                                            remask=remask)

    def uv_range_at_time(self, time, *, remask=False):
        """The `(u,v)` range of spatial pixels in the data array observed at the specified
        time.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.

        Returns:
            tuple[Pair, Pair]: `(uv_min, uv_max)`, where `uv_min` is the lower corner of
            the `(u,v)` rectangle observed and `uv_max` is the upper corner.
        """

        if self._time_is_1d:
            return self.uv_range_at_time_1d(time, self.uv_shape,
                                            axis=self._t_uv_axis, remask=remask)
        else:
            return self.uv_range_at_time_2d(time, self.uv_shape,
                                            slow=(1-self._fast_t_uv_axis),
                                            fast=self._fast_t_uv_axis,
                                            remask=remask)

    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Parameters:
            dtime (float): The time offset to apply to the observation, in units of
                seconds. A positive value shifts the observation later.

        Returns:
            Observation: A (shallow) copy of the object with a new time.
        """

        return TimedImage(self._axes, self.cadence.time_shift(dtime), fov=self.fov,
                          path=self.path, frame=self.frame, **self.subfields)

    def inventory(self, bodies, **kwargs):
        """Info about the bodies that appear unobscured inside the FOV.

        This operates by returning every body that would have been inside the FOV of this
        observation if it were instead a Snapshot, evaluated at the given `tfrac`. As a
        result, the time-dependence of the pixels is ignored; use at your own risk.

        Parameters:
            bodies (list): The names of the body objects to be included in the inventory.
            **kwargs: The remaining keyword options of Snapshot.inventory(), which see.

        Returns:
            (list, array, or dict): As described in Snapshot.inventory().

        Raises:
            NotImplementedError: If the cadence extends this observation beyond the shape
                of its FOV.
        """

        # TODO
        if self._extended_fov:
            raise NotImplementedError('inventory is not implemented for '
                                      'TimedImage with cadence-extended FOV')

        return self._snapshot.inventory(bodies, **kwargs)

##########################################################################################
