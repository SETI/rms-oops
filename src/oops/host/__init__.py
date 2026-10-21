##########################################################################################
# oops/host/__init__.py
##########################################################################################

from ._errors       import (HostError, HostFormatError, HostIndexError, PdsHostError,
                            UnknownHostError, VicarHostError, WrongHostError)
from .              import host_
from .host_         import Host
from ._fits_support import get_fits_image_hdu
from ._pds3_support import read_pds3_image_array

# Assigned rather than imported so that the doc comment, which autodoc reads only from the
# module it documents, describes the name where it is published.
#: The Observations read from one file or index row when there is more than one, as a
#: tuple in which an entry is None where a selected Observation is absent, or as a NumPy
#: object array of Observations.
ObsArray = host_.ObsArray

__all__ = ['Host', 'ObsArray', 'get_fits_image_hdu', 'read_pds3_image_array',
           'HostError', 'HostFormatError', 'HostIndexError', 'PdsHostError',
           'UnknownHostError', 'VicarHostError', 'WrongHostError']

##########################################################################################
