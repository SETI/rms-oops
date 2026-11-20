##########################################################################################
# oops/host/__init__.py
##########################################################################################

from ._errors       import (AmbiguousHostError, HostError, HostFormatError,
                            HostIndexError, PdsHostError, UnknownHostError,
                            VicarHostError, WrongHostError)
from .              import _pds3_support, host_
from .host_         import Host
from ._fits_support import get_fits_image_hdu
from ._pds3_support import is_pds3_null, read_pds3_image_array

# Assigned rather than imported so that the doc comment, which autodoc reads only from the
# module it documents, describes the name where it is published.
#: The Observations read from one file or index row when there is more than one, as a
#: tuple in which an entry is None where a selected Observation is absent, or as a NumPy
#: object array of Observations.
ObsArray = host_.ObsArray

#: The texts that stand for a missing value in a PDS3 label, an index, or a VICAR header,
#: in upper case and without padding: the PDS3 null constants "UNK", "N/A", and "NULL",
#: and the placeholders "UNKNOWN", "NOT APPLICABLE", "---", and "". See
#: :func:`is_pds3_null`.
PDS3_NULL_TEXTS = _pds3_support.PDS3_NULL_TEXTS

__all__ = ['Host', 'ObsArray', 'PDS3_NULL_TEXTS', 'get_fits_image_hdu', 'is_pds3_null',
           'read_pds3_image_array',
           'AmbiguousHostError', 'HostError', 'HostFormatError', 'HostIndexError',
           'PdsHostError', 'UnknownHostError', 'VicarHostError', 'WrongHostError']

##########################################################################################
