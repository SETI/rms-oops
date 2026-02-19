##########################################################################################
# oops/frame/spicetype1frame.py: Subclass SpiceType1Frame of Frame
##########################################################################################

import numpy as np
import cspyce

from polymath              import Matrix3, Scalar, Vector3
from oops.cache            import Cache
from oops.frame            import Frame, LinkedFrame
from oops.frame.spiceframe import SpiceFrame
from oops.transform        import Transform


class SpiceType1Frame(SpiceFrame):
    """A Frame object defined within the SPICE toolkit as a Type 1 (discrete) C kernel."""

    _FRAME_LOOKUP = {}          # (code, reference, tick_tolerance) -> SpiceFrame

    def __init__(self, spice_frame, tick_tolerance, reference=None, *, frame_id=None,
                 cache_size=100):
        """Constructor for a SpiceType1Frame.

        Parameters:
            spice_frame (str or int): The name, frame code, or frame name as used in the
                SPICE toolkit.
            tick_tolerance (float, int or str): A number or string defining the time
                tolerance in spacecraft clock ticks for the Frame returned.
            reference (SpiceFrame or str, optional): The Frame or ID of the Frame relative
                to which this frame is defined. This must be a SpiceFrame or else, by
                default, J2000.
            frame_id (str, optional): The ID under which to register this Frame. If not
                specified, the name as defined in the SPICE Toolkit is used. Note that
                SpiceType1Frames are always registered.
            cache_size (int, optional): Number of transforms to cache. This can be useful
                because it avoids unnecessary SPICE calls when the Frame is being used
                repeatedly at a finite set of times.

        Raises:
            LookupError: If `spice_frame` is not a recognized frame name or frame code
                within the SPICE Toolkit.
            ValueError: If `reference` is not a SpiceFrame or J2000.
        """

        self._fill_spice_info(spice_frame, reference)

        # Fill in the time tolerance
        if isinstance(tick_tolerance, str):
            self._tick_tolerance = cspyce.sctiks(self._spice_origin_code, tick_tolerance)
        else:
            self._tick_tolerance = tick_tolerance

        self._time_tolerance = None             # filled in on first use

        self._cache_size = cache_size or 100

        # If the reference is not J2000, construct the primary version first
        if self._reference != Frame.J2000:
            wrt_j2000 = SpiceType1Frame.get(self._spice_frame_name, self._tick_tolerance,
                                            Frame.J2000, frame_id=frame_id,
                                            cache_size=self._cache_size)
            # Cache but don't register under this frame ID
            self._register(frame_id=None)
            self._wayframe = wrt_j2000._wayframe
            self._frame_id = wrt_j2000._frame_id
        else:
            # If the reference is J2000, register as normal
            _ = SpiceFrame._FOR_CODE.setdefault(self._spice_frame_name, self)
            self._register(frame_id or self._spice_frame_name.replace(' ', '_'))

        self._refresh()
        self._register(frame_id or self._spice_frame_name)

        # Save for use by get()
        for cache_size in (self._cache_size, None):
            key = (self._spice_frame_name, reference, self._tick_tolerance, cache_size)
            _ = SpiceFrame._FRAME_LOOKUP.setdefault(key, self)

    def _refresh(self):
        self._cache = Cache(self._cache_size)   # saves result for multiple single times

        self._cached_transform = None           # saves result for one shaped time
        self._cached_time = None
        self._cached_shape = None

        if hasattr(self, '_quickframes'):
            self._quickframes.clear()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        return (self._spice_frame_name, self._reference, self._tick_tolerance,
                self.stripped_id, self._cache_size, self._get_quickframes())

    def __setstate__(self, state):
        (frame_name, reference, tick_tolerance, frame_id, cache_size, quickframes) = state
        self.__init__(frame_name, reference, tick_tolerance=tick_tolerance,
                      frame_id=frame_id, cache_size=cache_size)
        if quickframes:
            self._quickframes = quickframes

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, quick=None):
        """A Transform that rotates from the reference frame into this frame.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.
        """

        # Fill in the time tolerance in seconds
        if self._time_tolerance is None:
            time = Scalar.as_scalar(time)
            ticks = cspyce.sce2c(self._spice_body_code, time.vals)
            ticks_per_sec = cspyce.sce2c(self._spice_body_code, time.vals + 1.) - ticks
            self._time_tolerance = self._tick_tolerance / ticks_per_sec

        # A single input time can be handled quickly
        time = Scalar.as_scalar(time)
        if time.shape == ():
            # Check cache first
            xform = self._cache[time.vals]
            if xform:
                return xform

            ticks = cspyce.sce2c(self._spice_body_code, time.vals)
            (matrix3, true_ticks) = cspyce.ckgp(self._spice_frame_code, ticks,
                                                self._tick_tolerance,
                                                self._spice_reference_name)
            xform = Transform(matrix3, Vector3.ZERO, self, self._reference)
            self._cache[time.vals] = xform
            return xform

        # Check to see if the latest shaped transform is adequate
        if np.shape(time.vals) == self._cached_shape:
            diff = np.abs(time.vals - self._cached_time)
            if np.all(diff < self._time_tolerance):
                return self._cached_transform

        # If all the times are close, we can return more quickly
        time_min = time.vals.min()
        time_max = time.vals.max()
        if (time_max - time_min) < self._time_tolerance:
            tick = cspyce.sce2c(self._spice_body_code, (time_min + time_max)/2.)
            (matrix3, true_tick) = cspyce.ckgp(self._spice_frame_code, tick,
                                               self._tick_tolerance,
                                               self._spice_reference_name)
            true_time = cspyce.sct2e(self._spice_body_code, true_tick)

            self._cached_shape = time.shape
            self._cached_time = true_time
            self._cached_transform = Transform(matrix3, Vector3.ZERO, self,
                                               self._reference)
            return self._cached_transform

        # Otherwise, process the array...
        ticks = cspyce.sce2c_vector(self._spice_body_code, time.vals.ravel())
        matrix3, true_ticks = cspyce.ckgp_vector(self._spice_frame_code, ticks,
                                                 self._tick_tolerance,
                                                 self._spice_reference_name)
        matrix3 = Matrix3.as_matrix3(matrix3).reshape(time.shape)
        true_times = cspyce.sct2e_vector(self._spice_body_code, true_ticks)

        self._cached_shape = time.shape
        self._cached_time = true_times
        self._cached_transform = Transform(matrix3, Vector3.ZERO, self, self._reference)
        return self._cached_transform

    def transform_at_time_if_possible(self, time, quick=None):
        """Transform that rotates coordinates from the reference frame to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Unlike method `transform_at_time`, this variant tolerates times that raise cspyce
        errors. It returns a new time Scalar along with the new Transform, where both
        objects skip over the times at which the transform could not be evaluated.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (tuple): The tuple (`newtimes`, `transform`), where:

            * `newtimes` (Scalar): Times at which `transform` has been provided; this may
              be a subset of the input times given because it omits times at which the
              Transform could not be evaluated.
            * `transform` (Transform): The Tranform applicable at `newtimmes`. It rotates
              vectors from the reference frame to this frame.
        """

        # Fill in the time tolerance in seconds
        if self._time_tolerance is None:
            time = Scalar.as_scalar(time)
            ticks = cspyce.sce2c(self._spice_body_code, time.vals)
            ticks_per_sec = cspyce.sce2c(self._spice_body_code, time.vals + 1.) - ticks
            self._time_tolerance = self._tick_tolerance / ticks_per_sec

        # A single input time can be handled quickly
        time = Scalar.as_scalar(time)
        if time.shape == ():
            # Check cache first
            xform = self._cache[time.vals]
            if xform:
                return xform

            ticks = cspyce.sce2c(self._spice_body_code, time.vals)
            (matrix3, true_ticks) = cspyce.ckgp(self._spice_frame_code, ticks,
                                                self._tick_tolerance,
                                                self._spice_reference_name)
            xform = Transform(matrix3, Vector3.ZERO, self, self._reference)
            self._cache[time.vals] = xform
            return xform

        # Check to see if the latest shaped transform is adequate
        if np.shape(time.vals) == self._cached_shape:
            diff = np.abs(time.vals - self._cached_time)
            if np.all(diff < self._time_tolerance):
                return self._cached_transform

        # If all the times are close, we can return more quickly
        time_min = time.vals.min()
        time_max = time.vals.max()
        if (time_max - time_min) < self._time_tolerance:
            tick = cspyce.sce2c(self._spice_body_code, (time_min + time_max)/2.)
            (matrix3, true_tick) = cspyce.ckgp(self._spice_frame_code, tick,
                                               self._tick_tolerance,
                                               self._spice_reference_name)
            true_time = cspyce.sct2e(self._spice_body_code, true_tick)

            self._cached_shape = time.shape
            self._cached_time = true_time
            self._cached_transform = Transform(matrix3, Vector3.ZERO, self,
                                               self._reference)
            return self._cached_transform

        # Otherwise, process the array...
        ticks = cspyce.sce2c_vector(self._spice_body_code, time.vals.ravel())
        matrix3, true_ticks = cspyce.ckgp_vector(self._spice_frame_code, ticks,
                                                 self._tick_tolerance,
                                                 self._spice_reference_name)
        matrix3 = Matrix3.as_matrix3(matrix3).reshape(time.shape)
        true_times = cspyce.sct2e_vector(self._spice_body_code, true_ticks)

        self._cached_shape = time.shape
        self._cached_time = true_times
        self._cached_transform = Transform(matrix3, Vector3.ZERO, self, self._reference)
        return self._cached_transform

    ######################################################################################
    # SpiceFrame API
    ######################################################################################

    @staticmethod
    def get(spice_frame, tick_tolerance, reference=None, *, frame_id=None,
            cache_size=None):
        """The SpiceType1Frame defined by the given parameters.

        If a matching SpiceType1Frame already exists, it is returned; otherwise, a new one
        is constructed and returned.

        Parameters:
            spice_frame (str or int): The name, frame code, or frame name as used in the
                SPICE toolkit. Alternatively, an existing SpiceType1Frame (which might use
                the wrong reference frame).
            tick_tolerance (float, int or str, optional): A number or string defining the
                time  tolerance in spacecraft clock ticks for the Frame returned.
            reference (SpiceFrame or str, optional): The Frame or ID of the Frame relative
                to which this frame is defined. This must be a SpiceFrame or else, by
                default, J2000.
            frame_id (str, optional): The ID under which to register this Frame. If not
                specified, the name as defined in the SPICE Toolkit is used. Note that
                SpiceType1Frames are always registered. This input is used only if a new
                SpiceType1Frames is constructed; otherwise, the pre-existing ID is
                retained.
            cache_size (int, optional): Number of transforms to cache. This can be useful
                because it avoids unnecessary SPICE calls when the Frame is being used
                repeatedly at a finite set of times. If not specified, an existing
                SpiceType1Frames with any `cache_size` is returned.

        Raises:
            LookupError: If `spice_frame` is not a recognized frame name or frame code
                within the SPICE Toolkit.
            ValueError: If `reference` is not a SpiceFrame or J2000.
        """

        reference = Frame.as_wayframe(reference)

        # Handle a SpiceFrame input; use it if it matches
        if isinstance(spice_frame, SpiceType1Frame):
            if (reference == spice_frame._reference
                    and tick_tolerance == spice_frame._tick_tolerance
                    and cache_size in (spice_frame._cache, None)):
                return spice_frame
            # Otherwise, identify the name and continue
            name = spice_frame._spice_frame_name
        else:
            (_, name) = SpiceFrame._frame_code_and_name(spice_frame)

        # See if a pre-existing Frame matches the request (including ticks and cache size)
        key = (name, reference, tick_tolerance, cache_size)
        if key in SpiceFrame._FRAME_LOOKUP:
            return SpiceFrame._FRAME_LOOKUP[key]

        # Otherwise, we need a new SpiceType1Frame
        return SpiceType1Frame(name, tick_tolerance, reference,
                               frame_id=frame_id, cache_size=cache_size)

    def _get_shortcut(self, reference):
        """A Frame that directly transforms from the given reference to this
        SpiceType1Frame.

        This is an override of the default method, needed because the SPICE Toolkit can
        handle the connections between SpiceFrames very efficiently.

        Parameters:
            reference (Frame): The reference Frame, which must be a valid wayframe.
        """

        # Find the first SpiceFrame (or J2000) that's an ancestor of the reference
        ancestor = reference
        while not isinstance(ancestor, (SpiceFrame, Frame.J2000Frame)):
            ancestor = ancestor._reference

        # Get the SpiceType1Frame to the selected ancestor
        spice_frame = SpiceType1Frame.get(self, self._tick_tolerance, ancestor,
                                          cache_size=self._cache_size)

        # Maybe we're done
        if ancestor == reference:
            return spice_frame

        # Get the "remainder" frame from the ancestor to the reference, then link
        remainder = ancestor._wrt(reference, use_shortcuts=False)
        return LinkedFrame(spice_frame, remainder)

##########################################################################################

Frame._FRAME_SUBCLASSES.append(SpiceType1Frame)

##########################################################################################
