##########################################################################################
# oops/host/_errors.py
##########################################################################################

from pdsparser  import PdsError
from vicar      import VicarError

__all__ = ['HostError', 'HostFormatError', 'HostIndexError', 'PdsHostError',
           'UnknownHostError', 'VicarHostError', 'WrongHostError']


class HostError(ValueError):
    """Exception arising from the content of a data file, index, or label."""
    pass


class UnknownHostError(HostError):
    """If a data file is not associated with any registered Host subclass."""
    pass


class WrongHostError(HostError):
    """If a data file is not associated with a specified Host subclass."""
    pass


class HostIndexError(HostError):
    """Exception arising from the content of an index or its label."""
    pass


class HostFormatError(HostError):
    """If a data file is incorrectly formatted."""
    pass


class PdsHostError(HostFormatError, PdsError):
    """A subclass of both HostFormatError and PdsError."""
    pass


class VicarHostError(HostFormatError, VicarError):
    """A subclass of both HostFormatError and VicarError."""
    pass

##########################################################################################
