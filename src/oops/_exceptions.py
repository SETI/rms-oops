##########################################################################################
# oops/_exceptions.py
##########################################################################################
"""Exception classes raised by the `oops` library."""

import re

import julian

__all__ = ['OopsException', 'OopsIndexError', 'OopsKeyError', 'OopsRuntimeError',
           'OopsSpiceError', 'OopsTypeError', 'OopsValueError']

# The SPICE short error that opens the text of every exception cspyce raises, as in
#   SPICE(CKINSUFFDATA) -- sigerr -- insufficient C kernel data to evaluate ...
# A KeyError quotes its text when converted to a string, so the code is searched for
# rather than matched at the start.
_SPICE_CODE = re.compile(r'SPICE\([A-Z0-9_]+\)')


class OopsException(Exception):
    """Base class of every exception raised by `oops`."""

    pass


class OopsIndexError(IndexError, OopsException):
    """An index passed to an `oops` method is out of range."""

    pass


class OopsKeyError(KeyError, OopsException):
    """A key used to look up an `oops` object is not defined."""

    pass


class OopsRuntimeError(RuntimeError, OopsException):
    """An `oops` method encountered an internal consistency error."""

    pass


# cspyce raises a different builtin class for each kind of SPICE error (OSError for a
# missing C-kernel record from ckgp, RuntimeError for a missing frame connection, KeyError
# for an unknown name), so no one base matches them all. RuntimeError is the class cspyce
# raises for every SPICE error under its "RUNTIME" error action, and the one an oops
# SpiceFrame raises when the kernels lack the data for a time; the condition is a property
# of the kernels loaded at run time rather than of any argument value.
class OopsSpiceError(RuntimeError, OopsException):
    """The SPICE kernels cannot supply data that an `oops` operation needs.

    Raise it when a SPICE query fails because the loaded kernels do not hold what was
    asked for: no C-kernel record for the pointing at a time, no frame connection between
    two frames, no ephemeris for a body. It is not for a failure that has nothing to do
    with the content of the kernels, such as an invalid argument.

    One user is the host packages: a host raises this exception, and only this one, when
    SPICE cannot supply the pointing of an observation it is building. A host does not
    raise it when the caller supplies the pointing; see :class:`~oops.host.Host`.

    Build it with :meth:`from_spice_error` from the exception that cspyce raised, so that
    every caller produces the same message form, and raise it ``from`` that exception so
    that the full SPICE text remains available as its cause::

        try:
            frame.transform_at_time(time)
        except (LookupError, OSError, RuntimeError, ValueError) as err:
            raise OopsSpiceError.from_spice_error(err, description='no NAC pointing',
                                                  time=time, filepath=path) from err

    Parameters:
        message (str): The message of the exception.
        spice_code (str, optional): The SPICE short error, spelled as SPICE spells it,
            such as "SPICE(CKINSUFFDATA)".
        filepath (str | pathlib.Path | FCPath, optional): The file whose processing needed
            the SPICE data, such as the image file of an observation.

    Attributes:
        spice_code (str | None): The SPICE short error, such as "SPICE(CKINSUFFDATA)", or
            None if it is not known.
        filepath (str | pathlib.Path | FCPath | None): The file whose processing needed
            the SPICE data, or None if no file is involved or it is not known.
    """

    def __init__(self, message, *, spice_code=None, filepath=None):

        super().__init__(message)
        self.spice_code = spice_code
        self.filepath = filepath

    @classmethod
    def from_spice_error(cls, exc, *, description, time=None, filepath=None):
        """An exception describing a failed SPICE query, built from cspyce's exception.

        The message has the form "<description> at <time> for <filepath>: <spice_code>",
        in which each of the last three parts is left out, with the word or colon that
        introduces it, when it is None. The time is given as an ISO 8601 UTC time with
        three decimal places of seconds, such as "2010-01-01T00:00:00.000".

        The SPICE short error is the first text of the form "SPICE(NAME)", in which NAME
        consists of uppercase letters, digits and underscores, in the string conversion of
        `exc`. Each exception raised by cspyce opens with one. If `exc` has none, the
        `spice_code` attribute is None and the message ends after the time or the file.

        The exception is returned, not raised; the caller raises it ``from`` `exc`.

        Parameters:
            exc (BaseException): The exception raised by the SPICE query.
            description (str): What could not be obtained, such as "no Cassini ISS NAC
                pointing".
            time (float, optional): The time of the query in seconds TDB.
            filepath (str | pathlib.Path | FCPath, optional): The file whose processing
                needed the SPICE data.

        Returns:
            OopsSpiceError: An instance of the class on which the method is called, whose
            `spice_code` is the SPICE short error found in `exc` and whose `filepath` is
            `filepath`.
        """

        match = _SPICE_CODE.search(str(exc))
        spice_code = match[0] if match else None

        message = description
        if time is not None:
            message += ' at ' + julian.iso_from_tai(julian.tai_from_tdb(time), digits=3)
        if filepath is not None:
            message += f' for {filepath}'
        if spice_code is not None:
            message += f': {spice_code}'

        return cls(message, spice_code=spice_code, filepath=filepath)


class OopsTypeError(TypeError, OopsException):
    """A value passed to an `oops` method is not of the expected type."""

    pass


class OopsValueError(ValueError, OopsException):
    """A value passed to an `oops` method is not valid."""

    pass

##########################################################################################
