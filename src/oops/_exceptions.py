##########################################################################################
# oops/_exceptions.py
##########################################################################################
"""Exception classes raised by the `oops` library."""

__all__ = ['OopsException', 'OopsIndexError', 'OopsKeyError', 'OopsRuntimeError',
           'OopsTypeError', 'OopsValueError']


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


class OopsTypeError(TypeError, OopsException):
    """A value passed to an `oops` method is not of the expected type."""

    pass


class OopsValueError(ValueError, OopsException):
    """A value passed to an `oops` method is not valid."""

    pass

##########################################################################################
