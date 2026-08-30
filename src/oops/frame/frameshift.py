##########################################################################################
# oops/frame/frameshift.py: Subclass FrameShift of class Frame
##########################################################################################

from polymath      import Scalar
from oops.fittable import Fittable
from oops.frame    import Frame
import oops.mutable as mutable


class FrameShift(Frame, Fittable):
    """A Frame subclass defined by a time shift of another Frame."""

    _WAYFRAMES = {}

    def __init__(self, arg, /, frame, *, frame_id=None, freeze=False):
        """Constructor for a FrameShift.

        Parameters:
            arg (float, FrameShift, TimeShift, or PathShift): The initial time shift in
                seconds. Alternatively, if another time-shifted object is given, this
                object's time shift will always match that of the argument.
            frame (Frame or str): The Frame or the ID of the Frame to which the time shift
                applies.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_SHIFT" to the ID of `frame` (if it has
                an ID).
            freeze (bool, optional): True to return a frozen object; False to leave it
                unfrozen.

        Raises:
            KeyError: If `frame` is an ID string that has not been registered.
        """

        # Linking to a frozen object yields a frozen object
        if isinstance(arg, str):
            arg = Frame.as_frame(arg)
        if hasattr(arg, 'dt') and mutable.is_frozen(arg):
            freeze = True
            arg = arg.dt

        if hasattr(arg, 'dt'):
            self._link = arg
            self._dt = arg.dt       # _register() needs this before _refresh() runs
        else:
            self._dt = arg
            self._link = None

        self._frame = Frame.as_frame(frame)
        self._reference = self._frame._reference
        self._origin = self._reference._origin
        self._shape = self._frame._shape

        if frame_id == '+' and self._frame._frame_id:
            frame_id = self._frame._frame_id + '_SHIFT'

        self._register(frame_id)
        self.refresh()
        if freeze:
            self.freeze()

    def _wayframe_key(self):
        return (self._dt, self._frame, self._link)

    @property
    def dt(self):
        """The time shift in seconds applied to the Frame."""
        return self._dt

    @property
    def link(self):
        """The object to which this one is linked, or None if it is unlinked."""
        return self._link

    def _source(self):
        """The original source of the time shift if this object is linked to another;
        otherwise, self.
        """
        return self._link and self._link._source() or self

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        if hasattr(self._link, 'show'):
            return (f'{name}({self._link.show(level-1, skip)},\n'
                    f'{blanks}{self._frame.show(level-1, skip)})')
        if self._link is not None:
            return (f'{name}({self._link},\n'
                    f'{blanks}{self._frame.show(level-1, skip)})')
        head = f'{name}({self._dt}, '
        return head + self._frame.show(level-1, indent + len(head))

    ######################################################################################
    # Fittable interface
    ######################################################################################

    nparams = 1

    def _set_params(self, params):
        """Redefine the time offset of this FrameShift object.

        If this object is linked to another, the time offset of the linked object is also
        redefined.
        """

        if self._link:
            self._link.set_params(params)
            self._dt = self._link._dt
        else:
            self._dt = params[0]

    @property
    def params(self):
        """The fittable parameters of this FrameShift as a tuple of one time shift."""
        return (self._dt,)

    def _refresh(self):
        if self._link:
            self._dt = self._link._dt

    def _freeze(self):
        if self._link:
            self._dt = self._link._dt
            self._link = None
        self._reregister()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._dt, self._frame, self.stripped_id)

    def __setstate__(self, state):
        (dt, frame, frame_id) = state
        self.__init__(dt, frame, frame_id=frame_id)
        self.freeze()

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        time = Scalar.as_scalar(time)
        return self._frame.transform_at_time(time + self._dt, quick=quick)

##########################################################################################

Frame._FRAME_SUBCLASSES.append(FrameShift)

##########################################################################################
