##########################################################################################
# oops/frame/spiceerrorframe.py
##########################################################################################

from typing import TYPE_CHECKING

from polymath         import Scalar
from oops._exceptions import OopsSpiceError
from oops.frame       import Frame

if TYPE_CHECKING:
    import pathlib

    from filecache import FCPath


class SpiceErrorFrame(Frame):
    """A Frame that the SPICE kernels cannot supply, which raises when it is evaluated.

    A host gives this Frame to an observation whose pointing the SPICE kernels cannot
    supply when it builds the observation without checking the kernels, as its
    ``from_index`` does, so that the observation is still returned and the error is raised
    only when the pointing is used. Evaluating it raises :class:`~oops.OopsSpiceError`,
    built with :meth:`~oops.OopsSpiceError.from_spice_error` from the exception that the
    SPICE query raised and raised ``from`` that exception.

    The Frame is defined relative to J2000 and is never registered.
    """

    def __init__(self, cause, *, description, filepath=None):
        """Constructor for a SpiceErrorFrame.

        Parameters:
            cause (BaseException): The exception that the SPICE query raised.
            description (str): What could not be obtained, such as "no Cassini ISS NAC
                pointing".
            filepath (str | pathlib.Path | FCPath, optional): The file whose processing
                needed the SPICE data, such as the image file of the observation.
        """

        self._cause = cause
        self._description = description
        self._filepath = filepath

        self._reference = Frame.J2000
        self._origin = None
        self._shape = ()
        self._register()

    @property
    def cause(self) -> BaseException:
        """The exception that the SPICE query raised."""
        return self._cause

    @property
    def description(self) -> str:
        """What could not be obtained."""
        return self._description

    @property
    def filepath(self) -> 'str | pathlib.Path | FCPath | None':
        """The file whose processing needed the SPICE data; None if it is not known."""
        return self._filepath

    def transform_at_time(self, time, *, quick=None):
        """Raise the error of the SPICE data that is missing.

        Parameters:
            time (ScalarLike): The time in seconds TDB.
            quick (dict | bool, optional): Ignored.

        Raises:
            OopsSpiceError: Always. Its message names `time` if `time` is a single,
                unmasked value, and its cause is the exception that the SPICE query
                raised.
        """

        time = Scalar.as_scalar(time)
        single = time.shape == () and not time.mask
        raise OopsSpiceError.from_spice_error(self._cause,
                                              description=self._description,
                                              time=float(time.vals) if single else None,
                                              filepath=self._filepath) from self._cause

    def _show(self, level, indent=0):
        """The expanded description of this Frame used by :meth:`~oops.Frame.show`.

        Parameters:
            level (int): The number of levels of the Frame's definition to expand.
            indent (int, optional): The number of blanks by which to indent each line
                after the first.

        Returns:
            str: The description of this Frame.
        """

        return f'SpiceErrorFrame({self._description!r})'

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        return (self._cause, self._description, self._filepath)

    def __setstate__(self, state):
        (cause, description, filepath) = state
        self.__init__(cause, description=description, filepath=filepath)

##########################################################################################
