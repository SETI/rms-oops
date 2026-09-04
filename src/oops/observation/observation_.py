##########################################################################################
# oops/observation/observation_.py
##########################################################################################

import numpy as np
import numbers

from polymath              import Matrix3, Scalar, Pair, Vector, Vector3, Qube
from oops                  import mutable
from oops.config           import LOGGING, PATH_PHOTONS
from oops.event            import Event
from oops.frame            import Frame
from oops.frame.cmatrix    import Cmatrix
from oops.frame.navigation import Navigation
from oops.meshgrid         import Meshgrid
from oops.mutable          import Mutable


class Observation(Mutable):
    """An abstract class defining the timing and pointing of a data array's samples.

    The axes of an observation are related to up to two spatial axes and one time axis.
    Spatial axes *(u,v)* are defined within an :class:`~oops.FOV` object, and the timing
    is defined by a :class:`~oops.Cadence`. Time is specified as an offset in seconds
    relative to the start time of the observation. An observation provides methods to
    convert between the indices of the data array and the coordinates *(u,v,t)* that
    define a line of sight at a particular time.

    When indices have non-integer values, the integer part identifies one "corner" of the
    sample, and the fractional part locates a point within the sample, i.e., part way from
    the start time to the end time of an integration, or a location inside the boundaries
    of a spatial pixel. Half-integer indices fall at the midpoint of each sample.

    Attributes:
        cadence (Cadence): Defines the timing of the Observation.
        time (tuple or Pair): The start time and end time of the Observation overall, in
            seconds TDB. Inherited from `cadence`.
        midtime (float): The mid-time of the Observation, in seconds TDB. Inherited from
            `cadence`.
        fov (FOV): The field of view, which describes the field of view including any
            spatial distortion. It maps between spatial coordinates *(u,v)* and instrument
            coordinates *(x,y)*.
        uv_shape (tuple): The 2-D shape of the spatial axes of the data array, in *(u,v)*
            order. This differs from `fov.uv_shape` in cases where the time-dependence
            introduces an extra dimension.
        u_axis (int): The axis of the data array associated with the *u*-axis; -1 if that
            axis is not associated with an array index.
        v_axis (int): The axis of the data array associated with the *v*-axis; -1 if that
            axis is not associated with an array index.
        swap_uv (bool): True if the *v*-axis comes before the *u*-axis; False otherwise.
        t_axis (int or list): The axes of the data array associated with time. When a list
            has multiple values, this is the sequence of array indices that break down
            time into finer and finer divisions, ordered from left to right. Use -1 if the
            observation has no time-dependence.
        shape (list or tuple): The overall shape of the observation data. Where the size
            of an axis is unknown, e.g., for a wavelength axis, the value can be zero.
        path (Path): The path waypoint co-located with the instrument.
        frame (Frame): The wayframe of a coordinate frame fixed to the optics of the
            instrument. This frame has its *z*-axis pointing outward near the center of
            the line of sight, with the *x*-axis pointing rightward and the *y*-axis
            pointing downward.
        subfields (dict): All of the optional attributes. Additional subfields may be
            included as needed.
        data (numpy.ndarray): This subfield is reserved to contain the array of data from
            the Observation.
    """

    _INVENTORY_IMPLEMENTED = False

    _DEBUG = False      # True to log iterative convergence steps

    ######################################################################################
    # Methods to be defined for each subclass
    ######################################################################################

    def __init__(self):
        """Constructor for an Observation; each subclass defines its own."""

        pass

    @property
    def time(self):
        """The start and stop times of the observation overall, in seconds TDB."""

        return self.cadence.time

    @property
    def midtime(self):
        """The mid-time of the observation, in seconds TDB."""

        return self.cadence.midtime

    def uvt(self, indices, *, remask=False, derivs=True):
        """Coordinates *(u,v)* and time *t* for indices into the data array.

        This method supports non-integer index values.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.
            derivs (bool, optional): True to include derivatives in the returned values.

        Returns:
            tuple[Pair, Scalar]: The *(u,v)* location and the time in seconds TDB
            associated with the array `indices`.
        """

        raise NotImplementedError(f'{type(self).__name__}.uvt is not implemented')

    def uvt_range(self, indices, *, remask=False):
        """Ranges of *(u,v)* spatial coordinates and time for integer array indices.

        Parameters:
            indices (Scalar or Vector): Array indices.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Pair, Pair, Scalar, Scalar]: The lower and upper *(u,v)* corners of the
            detector, followed by the earliest and latest time TDB, associated with the
            given array `indices`.
        """

        raise NotImplementedError(f'{type(self).__name__}.uvt_range is not implemented')

    def time_range_at_uv(self, uv_pair, *, remask=False):
        """The start and stop times of the specified spatial pixel *(u,v)*.

        Parameters:
            uv_pair (Pair): Spatial *(u,v)* data array coordinates, truncated to integers
                if necessary.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: The start time and stop time in seconds TDB for each
            `uv_pair`.
        """

        raise NotImplementedError(f'{type(self).__name__}.time_range_at_uv is not '
                                  'implemented')

    def _time_range_at_uv_0d(self, uv_pair, *, remask=False):
        """The time range at *(u,v)* when the spatial and time axes are independent.

        Parameters:
            uv_pair (Pair): Spatial *(u,v)* data array coordinates, truncated to integers
                if necessary.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: The start time and stop time in seconds TDB for each
            `uv_pair`.
        """

        time_min = Scalar(self.time[0])     # shapeless scalars
        time_max = Scalar(self.time[1])

        if remask:
            uv_pair = Pair.as_pair(uv_pair, recursive=False)
            new_mask = self.fov.uv_is_outside(uv_pair)
            if new_mask.any_true_or_masked():
                new_mask = Qube.or_(new_mask.vals, new_mask.mask)
                time_min = Scalar.filled(uv_pair.shape, self.time[0], mask=new_mask)
                time_max = Scalar.filled(uv_pair.shape, self.time[1], mask=new_mask)

        return (time_min, time_max)

    def _time_range_at_uv_1d(self, uv_pair, *, axis=0, remask=False):
        """The time range at *(u,v)* for an observation with a 1-D cadence.

        Parameters:
            uv_pair (Pair): Spatial *(u,v)* data array coordinates, truncated to integers
                if necessary.
            axis (int, optional): 0 or 1, indicating the uv axis associated with the
                cadence.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: The start time and stop time in seconds TDB for each
            `uv_pair`.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        tstep = uv_pair.to_scalar(axis)

        # Re-mask the time-independent axis if necessary
        if remask:
            not_t_vals = uv_pair.vals[..., 1-axis]
            not_t_max = self.uv_shape[1-axis]
            new_mask = Qube.or_(not_t_vals < 0, not_t_vals > not_t_max)
            tstep = tstep.remask_or(new_mask)

        return self.cadence.time_range_at_tstep(tstep, remask=remask)

    def _time_range_at_uv_2d(self, uv_pair, *, fast=1, remask=False):
        """The time range at *(u,v)* for an observation with a 2-D cadence.

        Parameters:
            uv_pair (Pair): Spatial *(u,v)* data array coordinates, truncated to integers
                if necessary.
            fast (int, optional): 0 or 1, indicating the uv axis associated with the fast
                index of the cadence. The slow index is always 1 - fast.
            remask (bool, optional): True to mask values outside the field of view.

        Returns:
            tuple[Scalar, Scalar]: The start time and stop time in seconds TDB for each
            `uv_pair`.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=False)

        if fast == 1:
            return self.cadence.time_range_at_tstep(uv_pair, remask=remask)
        else:
            return self.cadence.time_range_at_tstep(uv_pair.swapxy(), remask=remask)

    def uv_range_at_time(self, time, *, remask=False):
        """The *(u,v)* range of spatial pixels observed at a specified time.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `time`.
        """

        raise NotImplementedError(f'{type(self).__name__}.uv_range_at_time is not ' +
                                  'implemented')

    def _uv_range_at_time_0d(self, time, uv_shape, *, remask=False):
        """The *(u,v)* range at a time, with time decoupled from the spatial axes.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            uv_shape (tuple or Pair): Shape of the active detector(s) within the FOV,
                in *(u,v)* order.
            remask (bool, optional): True to mask times that are out of range.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `time`.
        """

        # Without re-masking, shapeless Pairs are OK
        if not remask:
            return (Pair.INT00, Pair.as_pair(uv_shape))

        # Define the new mask
        time = Scalar.as_scalar(time, recursive=False)
        new_mask = Qube.or_(time.mask, self.cadence.time_is_outside(time).vals)

        # Without any mask, shapeless Pairs are OK
        if not np.any(new_mask):
            return (Pair.INT00, Pair.as_pair(uv_shape))

        # Construct the array of results if necessary
        uv_min = Pair.zeros(time.shape, dtype='int', mask=new_mask)
        return (uv_min, uv_min + Pair.as_pair(uv_shape))

    def _uv_range_at_time_1d(self, time, uv_shape, *, axis=0, remask=False):
        """The *(u,v)* range at a time, for an observation with a 1-D cadence.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            uv_shape (tuple): Shape of the active detector(s) within the FOV, in *(u,v)*
                order.
            axis (int, optional): 0 or 1, indicating the uv axis associated with the
                cadence. Alternatively, -1 indicates that the time axis is not associated
                with a spatial axis.
            remask (bool, optional): True to mask times that are out of range.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `time`.
        """

        if axis < 0:
            return self._uv_range_at_time_0d(time, uv_shape, remask=remask)

        (tstep_min,
         tstep_max) = self.cadence.tstep_range_at_time(time, remask=remask)

        uv_min_vals = np.zeros(tstep_min.shape + (2,), dtype='int')
        uv_max_vals = np.empty(tstep_min.shape + (2,), dtype='int')

        uv_min_vals[..., axis] = tstep_min.vals
        uv_max_vals[..., axis] = tstep_max.vals
        uv_max_vals[..., 1-axis] = uv_shape[1-axis]

        uv_min = Pair(uv_min_vals, tstep_min.mask)
        uv_max = Pair(uv_max_vals, tstep_min.mask)
        return (uv_min, uv_max)

    def _uv_range_at_time_2d(self, time, uv_shape, *, slow=0, fast=1, remask=False):
        """The *(u,v)* range at a time, for an observation with a 2-D cadence.

        Parameters:
            time (Scalar): Time values in seconds TDB.
            uv_shape (tuple): Shape of the active detector(s) within the FOV, in *(u,v)*
                order.
            slow (int, optional): 0 or 1, indicating the uv axis associated with the slow
                index of the cadence. Alternatively, -1 indicates that this index is not
                associated with a spatial axis.
            fast (int, optional): 0 or 1, indicating the uv axis associated with the fast
                index of the cadence. Alternatively, -1 indicates that this index is not
                associated with a spatial axis.
            remask (bool, optional): True to mask times that are out of range.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `time`.
        """

        (tstep_min,
         tstep_max) = self.cadence.tstep_range_at_time(time, remask=remask)

        if slow == 0 and fast == 1:
            return (tstep_min, tstep_max)
        elif slow == 1 and fast == 0:
            return (tstep_min.swapxy(), tstep_max.swapxy())

        uv_min_vals = np.zeros(tstep_min.shape + (2,), dtype='int')
        uv_max_vals = np.empty(tstep_min.shape + (2,), dtype='int')
        uv_max_vals[..., 0] = uv_shape[0]
        uv_max_vals[..., 1] = uv_shape[1]

        if slow >= 0:
            uv_min_vals[..., slow] = tstep_min.vals[..., 0]
            uv_max_vals[..., slow] = tstep_max.vals[..., 0]
        if fast >= 0:
            uv_min_vals[..., fast] = tstep_min.vals[..., 1]
            uv_max_vals[..., fast] = tstep_max.vals[..., 1]

        uv_min = Pair(uv_min_vals, tstep_min.mask)
        uv_max = Pair(uv_max_vals, tstep_min.mask)
        return (uv_min, uv_max)

    def uv_range_at_tstep(self, tstep, *, remask=False):
        """The range of spatial *(u,v)* pixels active at a particular time step.

        Parameters:
            tstep (Scalar or Pair): Time step index. This is a Scalar for an observation
                with a 1-D cadence and a Pair for one with a 2-D cadence.
            remask (bool, optional): True to mask time steps outside the cadence.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `tstep`.
        """

        raise NotImplementedError(f'{type(self).__name__}.uv_range_at_tstep is not '
                                  'implemented')

    def _uv_range_at_tstep_0d(self, tstep, uv_shape, *, remask=False):
        """The *(u,v)* range at a time step, with time decoupled from the spatial axes.

        Every pixel is active at every time step, so the range always covers the whole
        field of view.

        Parameters:
            tstep (Scalar): Time step index.
            uv_shape (tuple or Pair): Shape of the active detector(s) within the FOV, in
                *(u,v)* order.
            remask (bool, optional): True to mask time steps outside the cadence.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `tstep`.
        """

        uv_min = Pair.INT00     # without a mask, return shapeless pairs
        uv_max = Pair.as_pair(uv_shape)

        # If the object needs a mask, expand it and mask it
        tstep = Scalar.as_scalar(tstep, recursive=False)
        if remask or np.any(tstep.mask):
            is_outside = (tstep.vals < 0) | (tstep.vals > self.cadence.shape[0])
            new_mask = Qube.or_(tstep.mask, is_outside)
            uv_min = Pair.zeros(tstep.shape, dtype='int', mask=new_mask)
            uv_max = Pair.filled(tstep.shape, uv_shape, mask=new_mask)

        return (uv_min, uv_max)

    def _uv_range_at_tstep_1d(self, tstep, uv_shape, *, axis=0, remask=False):
        """The *(u,v)* range at a time step, for an observation with a 1-D cadence.

        Parameters:
            tstep (Scalar): Time step index.
            uv_shape (tuple): Shape of the active detector(s) within the FOV, in *(u,v)*
                order.
            axis (int, optional): 0 or 1, indicating the uv axis associated with the
                cadence. Alternatively, -1 indicates that the time axis is not associated
                with a spatial axis.
            remask (bool, optional): True to mask time steps outside the cadence.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `tstep`. In the 1-D case, one pixel is active
            along `axis`; the full extent of the FOV is active along the other axis.
        """

        if axis < 0:
            return self._uv_range_at_tstep_0d(tstep, uv_shape, remask=remask)

        tstep = Scalar.as_scalar(tstep, recursive=False)
        tstep_int = tstep.int(top=self.cadence.shape[0], remask=remask, clip=True)

        uv_min_vals = np.zeros(tstep_int.shape + (2,), dtype='int')
        uv_max_vals = np.empty(tstep_int.shape + (2,), dtype='int')

        uv_min_vals[..., axis] = tstep_int.vals
        uv_max_vals[..., axis] = tstep_int.vals + 1
        uv_max_vals[..., 1-axis] = uv_shape[1-axis]

        uv_min = Pair(uv_min_vals, tstep_int.mask)
        uv_max = Pair(uv_max_vals, tstep_int.mask)
        return (uv_min, uv_max)

    def _uv_range_at_tstep_2d(self, tstep, uv_shape, *, slow=0, fast=1, remask=False):
        """The *(u,v)* range at a time step, for an observation with a 2-D cadence.

        Parameters:
            tstep (Pair): Time step index, as (slow, fast).
            uv_shape (tuple): Shape of the active detector(s) within the FOV, in *(u,v)*
                order.
            slow (int, optional): 0 or 1, indicating the uv axis associated with the slow
                index of the cadence. Alternatively, -1 indicates that this index is not
                associated with a spatial axis.
            fast (int, optional): 0 or 1, indicating the uv axis associated with the fast
                index of the cadence. Alternatively, -1 indicates that this index is not
                associated with a spatial axis.
            remask (bool, optional): True to mask time steps outside the cadence.

        Returns:
            tuple[Pair, Pair]: The lower (inclusive) and upper (exclusive) corners of the
            observed *(u,v)* rectangle at `tstep`.
        """

        tstep = Pair.as_pair(tstep, recursive=False)
        tstep_int = tstep.int(self.cadence.shape, remask=remask, clip=True)

        if slow == 0 and fast == 1:
            return (tstep_int, tstep_int + Pair.INT11)
        elif slow == 1 and fast == 0:
            swapped = tstep_int.swapxy()
            return (swapped, swapped + Pair.INT11)

        uv_min_vals = np.zeros(tstep_int.shape + (2,), dtype='int')
        uv_max_vals = np.empty(tstep_int.shape + (2,), dtype='int')
        uv_max_vals[..., 0] = uv_shape[0]
        uv_max_vals[..., 1] = uv_shape[1]

        if slow >= 0:
            uv_min_vals[..., slow] = tstep_int.vals[..., 0]
            uv_max_vals[..., slow] = tstep_int.vals[..., 0] + 1
        if fast >= 0:
            uv_min_vals[..., fast] = tstep_int.vals[..., 1]
            uv_max_vals[..., fast] = tstep_int.vals[..., 1] + 1

        uv_min = Pair(uv_min_vals, tstep_int.mask)
        uv_max = Pair(uv_max_vals, tstep_int.mask)
        return (uv_min, uv_max)

    def time_shift(self, dtime):
        """A copy of the observation object with a time-shift.

        Parameters:
            dtime (float): The time offset to apply to the observation, in units of
                seconds. A positive value shifts the observation later.

        Returns:
            Observation: A (shallow) copy of the object with a new time.
        """

        raise NotImplementedError(f'{type(self).__name__}.time_shift is not implemented')

    def copy(self):
        """An independent copy of this observation.

        The data array, if any, is shared with the original rather than duplicated. The
        frame, path, FOV and cadence are shared as well; these are canonical objects,
        registered and shared throughout OOPS, and duplicating them would break the
        identities that the Frame and Path registries rely on. What the copy does not
        share is its own state: it gets a new subfield dictionary, so subfields can be
        inserted into or deleted from either observation without disturbing the other, and
        it carries none of the original's internal record of what has been modified.

        A Fittable sub-object, such as a :class:`~oops.frame.Navigation` frame, is the
        exception: it is duplicated, so that fitting one observation leaves the other
        unchanged. The object to which it applies is still shared.

        Returns:
            Observation: A copy of the object.
        """

        fittables = [name for name in mutable.mutable_names(self)
                     if mutable.is_fittable(self.__dict__[name])]

        obs = object.__new__(type(self))
        obs.__dict__ = self.__dict__.copy()
        obs.subfields = self.subfields.copy()

        # Anything that can be modified in place gets duplicated; a subfield is also an
        # attribute, so both records have to be updated
        for name in fittables:
            subobj = self.__dict__[name].copy()
            obs.__dict__[name] = subobj
            if name in obs.subfields:
                obs.subfields[name] = subobj

        # The copy is a new object, so it must not inherit the bookkeeping that describes
        # the original; otherwise the two would share one record of which sub-objects are
        # mutable and when each was last refreshed
        for key in [k for k in obs.__dict__ if k.startswith('_MUTABLE')]:
            del obs.__dict__[key]

        return obs

    def navigate(self, angles):
        """A copy of this Observation with a :class:`~oops.frame.Navigation` rotation
        applied.

        The rotation is defined by two or three rotation angles of a Navigation object.

        The copy is created by copy(), so it shares the data array with this observation
        but is given a Navigation frame of its own; re-pointing the copy leaves this
        observation unchanged.

        Parameters:
            angles (tuple or list): Two or three angles of rotation in radians. The
                order of the rotations is about the *y*, *x*, and (optionally) *z* axes.
                These angles rotate a vector in the reference frame into this frame.

        Returns:
            Observation: A new Observation with the navigation applied.
        """

        obs = self.copy()

        # Identify the non-navigated frame, so that repeated calls replace the Navigation
        # rather than stacking a second one on top of it
        if isinstance(obs.frame, Navigation):
            frame = obs.frame.reference
        else:
            frame = obs.frame

        obs.frame = Navigation(angles, reference=frame)
        mutable._invalidate(obs)
        mutable._increment(obs)
        return obs

    def set_frame(self, frame):
        """Replace the frame of this observation in place.

        This modifies the observation itself rather than returning a copy. Any object
        already holding a reference to this observation, such as a Backplane, sees the new
        frame but retains everything it derived from the old one; call
        :meth:`~oops.mutable.Mutable.refresh` on such an object afterward.

        Parameters:
            frame (Frame): The new frame.
        """

        # An observation that is not mutable reports itself as frozen because it has
        # nothing to freeze, so both tests are needed to single out an observation that
        # was frozen deliberately
        if mutable.is_mutable(self) and mutable.is_frozen(self):
            raise ValueError(f'{type(self).__name__} object is frozen')

        self.frame = frame

        # The new frame may be Fittable where the old one was not, or vice versa, so the
        # record of which sub-objects are mutable is now stale
        mutable._invalidate(self)
        mutable._increment(self)

    def get_spice_cmatrix(self, tstep=None, *, time=None):
        """The C matrix of this observation, in the convention used by the SPICE toolkit.

        This observation must carry a `spice_to_frame` subfield, the rotation from the
        SPICE frame convention of the instrument to the oops convention. It is inserted by
        the host module that created the observation.

        Parameters:
            tstep (float or Scalar): The time step index or sequence of time step index
                values, as interpreted by this Observation's cadence.
            time (float or Scalar): The time in seconds TDB during the Observation. Note
                that at most one of `tstep` and `time` can be specified; if neither is
                given, the midtime of this Observation is used.

        Returns:
            Matrix3: The rotation from J2000 coordinates into the SPICE frame of the
            instrument or host.
        """

        if not hasattr(self, 'spice_to_frame'):
            raise AttributeError(f'{type(self).__name__} does not have a '
                                 '"spice_to_frame" attribute')

        if time is None:
            if tstep is None:
                time = self.midtime
            else:
                time = self.cadence.time_at_tstep(tstep)
        elif tstep is not None:
            raise ValueError('tstep and time cannot both be specified')

        frame = self.frame.wrt(Frame.J2000)
        xform = frame.transform_at_time(time)
        return self.spice_to_frame.inverse() * xform.matrix

    def set_spice_cmatrix(self, matrix):
        """Set this Observation's frame from a SPICE C matrix.

        The matrix uses the convention of the SPICE toolkit's C kernel rather than the
        OOPS convention.

        This replaces the frame outright, so the observation is left with a fixed pointing
        relative to J2000.

        Parameters:
            matrix (Matrix3): The C matrix rotating J2000 coordinates into the SPICE frame
                of the instrument, as a Matrix3 or as anything that can be converted to
                one.
        """

        if not hasattr(self, 'spice_to_frame'):
            raise AttributeError(f'{type(self).__name__} does not have a '
                                 '"spice_to_frame" attribute')

        frame = Cmatrix(self.spice_to_frame * Matrix3.as_matrix3(matrix))
        self.set_frame(frame)

    ######################################################################################
    # Subfield support methods
    ######################################################################################

    def insert_subfield(self, key, value):
        """Insert a subfield into this observation, also making it an attribute.

        Parameters:
            key (str): The name of the subfield.
            value (Any): The value of the subfield.
        """

        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

    def delete_subfield(self, key):
        """Delete a subfield of this observation, if it is present.

        Parameters:
            key (str): The name of the subfield.
        """

        if key in self.subfields:
            del self.subfields[key]
            del self.__dict__[key]

    def delete_subfields(self):
        """Delete all the subfields of this observation."""

        for key in list(self.subfields):
            del self.subfields[key]
            del self.__dict__[key]

    ######################################################################################
    # Methods probably not requiring overrides
    ######################################################################################

    def uv_is_outside(self, uv_pair, *, inclusive=True):
        """A Boolean mask identifying coordinates outside the FOV.

        Parameters:
            uv_pair (Pair): *(u,v)* coordinates.
            inclusive (bool, optional): True to interpret coordinate values at the upper
                end of each range as inside the FOV; False to interpret them as outside.

        Returns:
            Boolean: True where the *(u,v)* coordinate falls outside the FOV.
        """

        # Interpret the *(u,v)* coordinates
        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        (u,v) = uv_pair.to_scalars()

        # Create the mask
        if inclusive:
            return (u.tvl_lt(0) | v.tvl_lt(0) | u.tvl_gt(self.uv_shape[0])
                                              | v.tvl_gt(self.uv_shape[1]))
        else:
            return (u.tvl_lt(0) | v.tvl_lt(0) | u.tvl_gt(self.uv_shape[0])
                                              | v.tvl_ge(self.uv_shape[1]))

    def midtime_at_uv(self, uv, *, tfrac=0.5):
        """The time at a specified fraction of the exposure of spatial pixel *(u,v)*.

        Parameters:
            uv (Pair): *(u,v)* coordinates.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where 0 refers to the beginning and 1 refers to the end. Default is 0.5,
                the mid-time.

        Returns:
            Scalar: The time within the integration at the specified `uv`, in seconds TDB.
        """

        (time0, time1) = self.time_range_at_uv(uv)
        return time0 + tfrac * (time1 - time0)

    def meshgrid(self, origin=None, *, undersample=1, oversample=1, limit=None,
                 center_uv=None, fov_kwargs=None):
        """A :class:`~oops.Meshgrid` shaped to broadcast to the observation's shape.

        This works like :meth:`Meshgrid.for_fov` except that the *(u,v)* axes are assigned
        their correct locations in the axis ordering of the observation.

        Parameters:
            origin (Pair, optional): A single value, tuple or Pair defining the origin of
                the grid. Default is to place the first sample in the middle of the first
                pixel, allowing for under- or oversampling.
            undersample (Pair, optional): A single value, tuple or Pair defining the
                magnitude of under-sampling to be performed. For example, a value of 2
                would cause the meshgrid to sample every other pixel along each axis.
            oversample (Pair, optional): A single value, tuple or Pair defining the
                magnitude of over-sampling to be performed. For example, a value of 2
                would create a 2x2 array of samples inside each pixel.
            limit (Pair, optional): A single value, tuple or Pair defining the upper
                limits of the meshgrid. By default, this is the shape of the FOV.
            center_uv (Pair, optional): Reference point at the center of the FOV; use None
                for the default, which depends on the origin and limit.
            fov_kwargs (dict, optional): Parameters passed to the FOV methods,
                containing parameters that might affect the properties of the FOV.

        Returns:
            Meshgrid: The desired Meshgrid.
        """

        return Meshgrid.for_shape(self.fov, self.shape, self.u_axis, self.v_axis,
                                  origin=origin, undersample=undersample,
                                  oversample=oversample, limit=limit, center_uv=center_uv,
                                  fov_kwargs=fov_kwargs)

    def timegrid(self, meshgrid, *, oversample=1, tfrac_limits=(0,1)):
        """A Scalar of times broadcastable with the shape of the given meshgrid.

        Parameters:
            meshgrid (Meshgrid): The meshgrid defining spatial sampling.
            oversample (int, optional): 1 to obtain one time sample per pixel; > 1 for
                finer sampling in time.
            tfrac_limits (tuple, optional): A pair of fractional time limits, interpreted
                in different ways depending on the observation's structure:

                * If this observation has no time-dependence, these are the fractional
                  time limits within the overall exposure duration.
                * If this observation has time-dependence that is entirely coupled to
                  spatial axes, these are the fractional time limits within each pixel's
                  individual exposure duration.
                * If this observation has time-dependence that is entirely decoupled from
                  the spatial axes, these are the start and end times relative to the time
                  limits of the defined cadence.

                A single number is interpreted as a pair of identical limits.

        Returns:
            Scalar: The desired times in seconds TDB.

        Raises:
            NotImplementedError: If this observation has a 2-D time-dependence with only
                one axis coupled to a spatial axis.
        """

        if isinstance(tfrac_limits, numbers.Number):
            tfrac_limits = (tfrac_limits, tfrac_limits)

        # Handle a time-independent observation
        if self.t_axis == -1:

            dt = self.time[1] - self.time[0]
            time0 = self.time[0] + tfrac_limits[0] * dt
            time1 = self.time[0] + tfrac_limits[1] * dt

            # One step implies midtime, which can be returned as a scalar
            if oversample == 1:
                return Scalar(0.5 * (time0 + time1))

            # Otherwise, uniform time steps between endpoints
            fracs = np.arange(oversample) / (oversample - 1.)
            times = time0 + fracs * (time1 - time0)

            # Time is on a leading axis
            tshape = times.shape + len(self.shape) * (1,)
            return Scalar.as_scalar(times.reshape(tshape))

        # Get times at each pixel in meshgrid
        (tstarts, tstops) = self.time_range_at_uv(meshgrid.uv)

        # Scale based on tfrac_limits
        time0 = tstarts + tfrac_limits[0] * (tstops - tstarts)
        time1 = tstarts + tfrac_limits[1] * (tstops - tstarts)

        # Handle 1-D case
        if isinstance(self.t_axis, numbers.Number):

            # Time aligns with u-axis or v-axis
            if self.t_axis in (self.u_axis, self.v_axis):

                # One time step implies midtime
                if oversample == 1:
                    return Scalar.as_scalar(0.5 * (time0 + time1))

                # Otherwise, uniform time steps on a leading axis
                fracs = np.arange(oversample) / (oversample - 1.)
                fracs = fracs.reshape(fracs.shape + len(self.shape) * (1,))
                return Scalar(time0 + fracs * (time1 - time0))

            # Otherwise time is along a unique axis
            tstep0 = tfrac_limits[0] * self.cadence.shape[0]
            tstep1 = tfrac_limits[1] * self.cadence.shape[0]
            tsteps = np.arange(tstep0, tstep1 + 1.e-10, 1./oversample)
            times = self.cadence.time_at_tstep(tsteps)

            shape_list = len(self.shape) * [1]
            shape_list[self.t_axis] = len(times)
            times = Scalar.as_scalar(times).reshape(tuple(shape_list))
            return times

        # Handle a 2-D observation
        if (self.t_axis[0] not in (self.u_axis, self.v_axis) or
            self.t_axis[1] not in (self.u_axis, self.v_axis)):
                raise NotImplementedError(f'Observation.timegrid not implemented for '
                                          f't axes {self.t_axis}, '
                                          f'u axis {self.u_axis}, v axis {self.v_axis}')

        # Time aligns with u-axis AND v-axis

        # One time step implies midtime
        if oversample == 1:
            return Scalar.as_scalar(0.5 * (time0 + time1))

        # Otherwise, uniform time steps on a leading axis
        fracs = np.arange(oversample) / (oversample - 1.)
        fracs = fracs.reshape(fracs.shape + len(self.shape) * (1,))
        return Scalar(time0 + fracs * (time1 - time0))

    def event_at_grid(self, meshgrid=None, *, tfrac=0.5, time=None):
        """A photon arrival event from directions defined by a meshgrid.

        Parameters:
            meshgrid (Meshgrid, optional): Object describing the sampling of the field of
                view.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where 0 refers to the beginning and 1 refers to the end of each sample's
                integration interval. The default is 0.5, referring to the midtime of each
                sample in the `meshgrid`.
            time (Scalar, optional): Optional Scalar of absolute time in seconds. Only one
                of `tfrac` and `time` can be specified.

        Returns:
            Event: The corresponding Event.
        """

        if time is None:
            time = self.midtime_at_uv(meshgrid.uv, tfrac=tfrac)

        event = Event(time, Vector3.ZERO, self.path, self.frame)

        # Insert the arrival directions
        event.neg_arr_ap = meshgrid.los(time)

        return event

    def gridless_event(self, meshgrid=None, *, tfrac=0.5, time=None, shapeless=False):
        """A photon arrival event irrespective of the direction.

        Parameters:
            meshgrid (Meshgrid, optional): Object describing the sampling of the field of
                view; None for a directionless observation. Here, it is only used to
                define the times if time is None.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where 0 refers to the beginning and 1 refers to the end of each sample's
                integration interval. The default is 0.5, referring to the midtime of each
                sample in the `meshgrid`.
            time (Scalar, optional): Scalar of optional absolute time in seconds.
            shapeless (bool, optional): True to return a shapeless event, referring to the
                mean of all the times.

        Returns:
            Event: The corresponding Event.
        """

        if time is None:
            if meshgrid is None:
                time = self.time[0] + tfrac * (self.time[1] - self.time[0])
            else:
                time = self.midtime_at_uv(meshgrid.uv, tfrac=tfrac)

        if shapeless:
            time = time.mean()

        return Event(time, Vector3.ZERO, self.path, self.frame)

    @staticmethod
    def _scalar_from_indices(indices, axis, *, derivs=True):
        """The selected Scalar from a set of indices.

        The indices can be a Scalar or Vector, a NumPy array, or a number.

        Parameters:
            indices (Scalar, Vector, numpy.ndarray, or number): Array indices.
            axis (int): The array axis to select; -1 if this axis is not associated with
                an array index.
            derivs (bool, optional): True to include derivatives in the returned value.

        Returns:
            Scalar: The selected Scalar; None if `axis` is negative.

        Raises:
            IndexError: If `indices` is a single number but `axis` is neither 0 nor -1.
        """

        if axis < 0:
            return None

        if isinstance(indices, (Scalar, Pair, Vector)):
            return indices.to_scalar(axis, recursive=derivs)

        if isinstance(indices, numbers.Real):
            if axis not in (0, -1):
                raise IndexError('index out of range: ' + str(indices))
            return Scalar(indices)

        indices = np.array(indices)

        # The meaning of the last axis in a Numpy array is ambiguous
        if indices.shape[-1] > axis:
            return Scalar(indices[..., axis])

        return Scalar(indices)                  # might fail; not our problem

    ######################################################################################
    # Geometry solvers
    ######################################################################################

    def uv_from_ra_and_dec(self, ra, dec, *, tfrac=0.5, time=None, apparent=True,
                           derivs=False, iters=2, quick=None):
        """Convert arbitrary scalars of RA and dec to FOV *(u,v)* coordinates.

        Parameters:
            ra (Scalar): J2000 right ascensions.
            dec (Scalar): J2000 declinations.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where 0 refers to the beginning and 1 refers to the end of each sample's
                integration interval. The default is 0.5, referring to the midtime of each
                sample in the `meshgrid`.
            time (Scalar, optional): Scalar of optional absolute time in seconds. Only one
                of tfrac and time can be specified.
            apparent (bool, optional): True to interpret the (RA,dec) values as apparent
                coordinates; False to interpret them as actual coordinates. Default is
                True.
            derivs (bool, optional): True to propagate derivatives of ra and dec through
                to derivatives of the returned *(u,v)* Pairs.
            iters (int, optional): Iterations to perform until convergence is reached.
                Two is the most that should ever be needed; Snapshot should override to
                one.
            quick (dict, optional): To override the configured default parameters for a
                :class:`~oops.path.QuickPath` or :class:`~oops.frame.QuickFrame`; False
                to disable the use of QuickPaths and QuickFrames. The default
                configuration is defined in ``config.py``.

        Returns:
            Pair: *(u,v)* coordinates.

        Notes:
            The only reasons for iteration are that the C-matrix and the velocity WRT the
            SSB could vary during the observation. I doubt this would ever be significant.
        """

        # Convert given (ra,dec) to line of sight in SSB/J2000 frame
        neg_arr_j2000 = Vector3.from_ra_dec_length(ra, dec, recursive=derivs)

        # Interpret the time
        if time is None:
            obs_time = self.time[0] + tfrac * (self.time[1] - self.time[0])

            # Require at least two iterations if tfrac != 0.5. A comparison against a
            # shapeless Scalar returns a Python bool, which has no all(), so the result
            # has to be converted back before it can be tested.
            is_midtime = Scalar.as_scalar(Scalar.as_scalar(tfrac) == 0.5)
            if not is_midtime.all():
                iters = max(2, iters)

        else:
            obs_time = time
            iters = 1

        # Iterate until *(u,v)* has converged
        uv = None
        for count in range(iters):

            # Define the photon arrival event
            obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)

            if apparent:
                obs_event.neg_arr_ap_j2000 = neg_arr_j2000
            else:
                obs_event.neg_arr_j2000 = neg_arr_j2000

            # Convert to FOV coordinates
            prev_uv = uv
            uv = self.fov.uv_from_los_t(obs_event.neg_arr_ap, time=obs_time,
                                        derivs=derivs)

            # If this is the last iteration, we're done
            if count + 1 == iters:
                break

            # Update the time
            (t0, t1) = self.time_range_at_uv(uv)
            obs_time = t0 + tfrac * (t1 - t0)

            # Stop at convergence
            if uv == prev_uv:
                break

        return uv

    def uv_from_path(self, path, *, tfrac=0.5, time=None, derivs=False, guess=None,
                     quick=None, converge=None):
        """The *(u,v)* indices of an object in the FOV, given its path.

        WARNING: this method is not well tested.

        Parameters:
            path (Path): Object.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where 0 refers to the beginning and 1 refers to the end of each sample's
                integration interval. The default is 0.5, referring to the midtime of each
                sample in the `meshgrid`.
            time (Scalar, optional): Scalar of optional absolute time in seconds. Only one
                of tfrac and time can be specified; the other must be None.
            derivs (bool, optional): True to propagate derivatives of the link time and
                position into the returned event.
            guess (Scalar, optional): An optional guess at the light travel time from the
                path to the event.
            quick (dict, optional): To override the configured default parameters for a
                :class:`~oops.path.QuickPath` or :class:`~oops.frame.QuickFrame`; False
                to disable the use of QuickPaths and QuickFrames. The default
                configuration is defined in ``config.py``.
            converge (dict, optional): Parameters to override the configured default
                convergence parameters. The default configuration is defined in
                ``config.py``.

        Returns:
            Pair: The *(u,v)* indices of the pixel in which the point was found. The path
            is evaluated at the mid-time of this pixel.

        Notes:
            This procedure assumes that movement along a path is very limited during the
            exposure time of an individual pixel. It could fail to converge if there is a
            large gap in timing between adjacent pixels at a time when the object is
            crossing that gap. However, even then, it should select roughly the correct
            location. It could also fail to converge during a fast slew.
        """

        # Assemble convergence parameters
        if converge:
            defaults = PATH_PHOTONS.__dict__.copy()
            defaults.update(converge)
            converge = defaults
        else:
            converge = PATH_PHOTONS.__dict__

        # Take a guess at the observation time
        converged = False
        if time is None:
            obs_time = self.time[0] + tfrac * (self.time[1] - self.time[0])
            iters = converge['max_iterations']
            dlt_precision = converge['dlt_precision']
            max_dt = 1.e99
        else:
            # In this case, no guessing is needed
            obs_time = time
            iters = 0
            converged = True

        for count in range(iters):

            # Locate the object in the field of view
            obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)
            (path_event,
             obs_event) = path.photon_to_event(obs_event, derivs=False, guess=guess,
                                               quick=quick, converge=converge)

            # Locate the object in the FOV frame
            uv = self.fov.uv_from_los_t(obs_event.neg_arr_ap, time=obs_event.time,
                                        derivs=derivs)

            # Update the observation time based on pixel midtime
            (t0, t1) = self.time_range_at_uv(uv)
            new_obs_time = t0 + tfrac * (t1 - t0)

            # Test for convergence
            prev_max_dt = max_dt
            max_dt = (new_obs_time - obs_time).abs().max(builtins=True, masked=-1.)
            obs_time = new_obs_time

            if LOGGING.observation_iterations or Observation._DEBUG:
                LOGGING.convergence('Observation.uv_from_path',
                                    f'iter={count+1}; change[s]={max_dt:.6g}')

            if max_dt <= dlt_precision:
                converged = True
                break

            if max_dt >= prev_max_dt:
                break

        if not converged:
            LOGGING.warn('Observation.uv_from_path did not converge: ',
                         f'iter={count+1}; change[s]={max_dt:.6g}')

        # Return the results
        obs_event = Event(obs_time, Vector3.ZERO, self.path, self.frame)
        (path_event,
         obs_event) = path.photon_to_event(obs_event, derivs=derivs, guess=guess,
                                           quick=quick, converge=converge)

        return self.fov.uv_from_los_t(obs_event.neg_arr_ap, time=obs_time, derivs=derivs)

    def uv_from_coords(self, surface, coords, *, tfrac=0.5, time=None, underside=False,
                       derivs=False, quick=None, converge=None):
        """The *(u,v)* indices of a surface point, given its coordinates.

        Parameters:
            surface (Surface): The Surface object.
            coords (tuple[Scalar, ...]): Two or three surface coordinates. The Scalars
                need not be the same shape, but must broadcast to the same shape.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where 0 refers to the beginning and 1 refers to the end of each sample's
                integration interval. The default is 0.5, referring to the midtime of each
                sample in the `meshgrid`.
            time (Scalar, optional): Scalar of optional absolute time in seconds. Only one
                of `tfrac` and `time` can be specified; the other must be None.
            underside (bool, optional): True for the underside of the surface (emission >
                90 degrees) to be unmasked.
            derivs (bool, optional): True to propagate derivatives of the link time and
                position into the returned event.
            quick (dict, optional): To override the configured default parameters for a
                :class:`~oops.path.QuickPath` or :class:`~oops.frame.QuickFrame`; False
                to disable the use of QuickPaths and QuickFrames. The default
                configuration is defined in ``config.py``.
            converge (dict, optional): Parameters to override the configured default
                convergence parameters. The default configuration is defined in
                ``config.py``.

        Returns:
            Pair: The *(u,v)* indices of the pixel in which the point was found.
        """

        raise NotImplementedError(f'{type(self).__name__}.uv_from_coords is not '
                                  'implemented')

    def inventory(self, bodies, *, tfrac=None, time=None, expand=0., return_type='list',
                  fov=None, quick=None, converge=None):
        """Info about the bodies that appear unobscured inside the FOV.

        Restrictions: All inventory calculations are performed at an observation time
        specified by `tfrac` or `time`. All bodies are assumed to be spherical.

        Parameters:
            bodies (list): The names of the body objects to be included in the inventory.
            tfrac (Scalar, optional): Scalar of fractional times during the exposure,
                where 0 refers to the beginning and 1 refers to the end of each sample's
                integration interval. The default is 0.5, referring to the midtime of each
                sample in the `meshgrid`.
            time (Scalar, optional): Scalar of optional absolute time in seconds TDB. Only
                one of `tfrac` and `time` can be specified.
            expand (float, optional): An optional angle in radians by which to extend the
                limits of the field of view. This can be used to accommodate pointing
                uncertainties.
            return_type (str, optional): One of "list", "flags", or "full":

                * "list": Return the inventory as a list of names.
                * "flags": Return the inventory as an array of boolean flag values in the
                  same order as bodies.
                * "full": Return the inventory as a dictionary of dictionaries. The main
                  dictionary is indexed by body name. The subdictionaries contain
                  attributes of the body in the FOV.

            fov (FOV, optional): Use this fov; if None, use `self.fov`.
            quick (dict, optional): To override the configured default parameters for a
                :class:`~oops.path.QuickPath` or :class:`~oops.frame.QuickFrame`; False
                to disable the use of QuickPaths and QuickFrames. The default
                configuration is defined in ``config.py``.
            converge (dict, optional): Parameters to override the configured default
                convergence parameters. The default configuration is defined in
                ``config.py``.

        Returns:
            list, numpy.ndarray, or dict: The inventory, in the form named by
            `return_type`.

            * If return_type is "list", it returns a list of the names of all the body
              objects that fall at least partially inside the FOV and are not completely
              obscured by another object in the list.

            * If return_type is "flags", it returns a boolean array containing True
              everywhere that the body falls at least partially inside the FOV and is not
              completely obscured.

            * If return_type is "full", it returns a dictionary with one entry per body,
              whether or not that body falls inside the FOV. Each dictionary entry is
              itself a dictionary containing data about the body in the FOV:

                - `"name"` (str): The body name.
                - `"inside"` (bool): True if the body is unobscured inside the FOV.
                - `"center_uv"` (ndarray): The *(u,v)* coordinates of the center
                  point, as two floats.
                - `"center"` (ndarray): The direction of the center point, as three
                  floats.
                - `"range"` (float): The distance in km.
                - `"outer_radius"` (float): The outer radius of the body in km.
                - `"inner_radius"` (float): The inner radius of the body in km.
                - `"resolution"` (ndarray): The resolution in the *(u,v)* directions
                  at the given range, as two floats.
                - `"u_min"` (int): The minimum `u` value covered by the body, clipped to
                  the FOV boundaries.
                - `"u_max"` (int): The maximum `u` value covered by the body, clipped to
                  the FOV boundaries.
                - `"v_min"` (int): The minimum `v` value covered by the body, clipped to
                  the FOV boundaries.
                - `"v_max"` (int): The maximum `v` value covered by the body, clipped to
                  the FOV boundaries.
                - `"u_min_unclipped"` (int): Same as "u_min", but not clipped.
                - `"u_max_unclipped"` (int): Same as "u_max", but not clipped.
                - `"v_min_unclipped"` (int): Same as "v_min", but not clipped.
                - `"v_max_unclipped"` (int): Same as "v_max", but not clipped.
                - `"u_pixel_size"` (float): The diameter of the body in pixels in units of
                  the *u* pixels.
                - `"v_pixel_size"` (float): The diameter of the body in pixels in units of
                  the *v* pixels.

        Raises:
            ValueError: If `return_type` is not one of "list", "flags", or "full".
        """

        raise NotImplementedError(f'{type(self).__name__}.inventory is not implemented')

    ######################################################################################
    # Support for parallel observations
    ######################################################################################

    def parallel_los(self, parallel, los, *, time=None, derivs=False):
        """The line of sight in a parallel observation's FOV.

        Parameters:
            parallel (Observation): A parallel observation (same origin and time,
                different frame and FOV).
            los (Vector3): A line of sight in this observation.
            time (Scalar, optional): Absolute time in seconds TDB; None to assume this
                observation's midtime.
            derivs (bool, optional): True to include the derivatives of the los in the
                result.

        Returns:
            Vector3: The same line of sight, expressed in the frame of the parallel
            observation.
        """

        # Define the relative frame (assuming a common origin)
        # This frame rotates vectors from this frame to the parallel frame.
        frame = self.frame.wrt(parallel.frame)

        # Convert the LOS to the frame of the parallel observation
        time = self.midtime if time is None else time
        xform = frame.transform_at_time(time)
        return xform.rotate(los, derivs=derivs)

    def parallel_uv(self, parallel, uv, *, time=None, derivs=False):
        """The *(u,v)* pixel coordinates in a parallel observation's FOV.

        Parameters:
            parallel (Observation): A parallel observation (same origin and time,
                different frame and FOV).
            uv (Pair): *(u,v)* pixel coordinates in this observation.
            time (Scalar, optional): Absolute time in seconds TDB; None to assume this
                observation's midtime.
            derivs (bool, optional): True to include the derivatives of `uv` in the
                result.

        Returns:
            Pair: The *(u,v)* coordinates.
        """

        # Convert the coordinates to a line of sight
        time = self.midtime if time is None else time
        los = self.fov.los_from_uvt(uv, time=time, derivs=derivs)

        # Transform to the parallel observation
        los = self.parallel_los(parallel, los, time=time, derivs=derivs)

        # Convert to coordinates in the new FOV
        return parallel.fov.uv_from_los_t(los, time=time, derivs=derivs)

    def parallel_offset_angles(self, parallel, angles, *, time=None):
        """The offset angles in a parallel observation's FOV and frame.

        Parameters:
            parallel (Observation): A parallel observation (same `origin` and `time`,
                different `frame` and `fov`). Alternatively, a tuple of two values:
                `(frame, fov)`.
            angles (tuple or list): Two offset angles in radians. The first rotation is
                about the *y* axis of this observation's frame and the second is about the
                *x* axis.
            time (Scalar, optional): Absolute time in seconds TDB; None to assume this
                observation's midtime.

        Returns:
            tuple[Scalar, Scalar]: The offset angles in radians about the *y* and *x* axes
            of the parallel observation's frame.
        """

        if isinstance(parallel, Observation):
            parallel_frame = parallel.frame
            parallel_fov = parallel.fov
        else:
            (parallel_frame, parallel_fov) = parallel

        # Define the relative frame (assuming a common origin)
        # This frame rotates vectors from this observation's frame to the parallel frame
        frame = self.frame.wrt(parallel_frame)
        time = self.midtime if time is None else time
        xform = frame.transform_at_time(time)

        # Get the parallel observation's line of sight in this frame
        uv = parallel_fov.uv_shape/2.
        los0_parallel = parallel_fov.los_from_uvt(uv, time=time)
        los0 = xform.unrotate(los0_parallel)

        # Perform the rotations in this frame
        # The angles refer to rotations of the axes, not the vectors, so they
        # need to be reversed here.
        los1 = los0.spin(Vector3.YAXIS, angles[0])
        los1 = los1.spin(Vector3.XAXIS, angles[1])

        # Convert back to the parallel's frame
        los1_parallel = xform.rotate(los1)

        # Return the new rotation angles
        return los0_parallel.offset_angles(los1_parallel)

    def parallel_offset_duv(self, parallel, duv, *, time=None, origin=None):
        """The *(u,v)* offset from the center of a parallel observation's FOV.

        Parameters:
            parallel (Observation): A parallel observation (same origin and time,
                different frame and FOV).
            duv (Pair): The *(u,v)* coordinate offset from the predicted location of a
                feature to its actual location.
            time (Scalar, optional): Absolute time in seconds TDB; None to assume this
                observation's midtime.
            origin (Pair, optional): The *(u,v)* coordinates of the reference point in
                this observation's FOV, from which the offset is measured. If unspecified,
                the center of the FOV is assumed.

        Returns:
            Pair: The pointing offset as a *(u,v)* pixel offset in the parallel
            observation's FOV.
        """

        time = self.midtime if time is None else time
        angles = self.fov.offset_angles_from_duv(duv, time=time, origin=origin)
        angles = self.parallel_offset_angles(parallel, angles, time=time)
        return parallel.fov.offset_duv_from_angles(angles, time=time)

##########################################################################################
