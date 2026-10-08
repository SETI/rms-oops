##########################################################################################
# oops/host/_errors.py
##########################################################################################

from pdsparser  import PdsError
from vicar      import VicarError

__all__ = ['HostError', 'HostFormatError', 'PdsHostError', 'UnknownHostError',
           'VicarHostError']


class HostError(OSError):
    """Exception arising from the content of a data file or label."""
    pass


class UnknownHostError(HostError):
    """If a data file is not associated with any registered Host subclass."""
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
