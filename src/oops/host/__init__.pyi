##########################################################################################
# oops/host/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.host`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.

The detectors and support methods that a Host subclass overrides or calls, such as
`_register` and `_detect_in_pds3`, are private and are not published here.
"""

import pathlib
from typing import Any, TypeAlias

from astropy.io.fits import HDUList, ImageHDU, PrimaryHDU
from filecache import FCPath
from numpy import ndarray
from pdsparser import Pds3Label, PdsError
from vicar import VicarError
from oops import Body
from oops.cadence import Cadence
from oops.calibration import Calibration
from oops.fov import FOV
from oops.frame import Frame
from oops.lightsource import LightSource
from oops.observation import Observation
from oops.path import Path

__all__ = ['Host', 'ObsArray', 'get_fits_image_hdu', 'read_pds3_image_array',
           'HostError', 'HostFormatError', 'HostIndexError', 'PdsHostError',
           'UnknownHostError', 'VicarHostError', 'WrongHostError']

ObsArray: TypeAlias = ndarray | tuple[Observation | None, ...]

class HostError(ValueError): ...
class UnknownHostError(HostError): ...
class WrongHostError(HostError): ...
class HostIndexError(HostError): ...
class HostFormatError(HostError): ...
class PdsHostError(HostFormatError, PdsError): ...
class VicarHostError(HostFormatError, VicarError): ...

class Host:
    NAME: str
    SUPPLEMENT: object
    @staticmethod
    def from_file(filepath: str | pathlib.Path | FCPath, *,
        pds3_options: dict[str, Any] | None = None,
        select: int | slice | str | tuple[int | slice | str, ...] | None = None,
        astrometry: bool = False, target: str | Body | None = None,
        lightsource: LightSource | None = None, path: str | Path | None = None,
        frame: str | Frame | None = None, cadence: Cadence | None = None,
        fov: FOV | None = None,
        calibrations: Calibration | list[Calibration] | None = ...,
        timeshift: tuple[float, str] | None = None,
        navigation: tuple[float, ...] | None = None, tracker: str | None = None,
        parallel: Observation | None = None,
        **kwargs: Any) -> Observation | ObsArray | None: ...
    @staticmethod
    def from_index(filepath: str | pathlib.Path | FCPath, *,
        supplement: str | pathlib.Path | FCPath | object | None = ...,
        label: Pds3Label | None = None, row_dicts: list[dict[str, Any]] | None = None,
        pds3_options: dict[str, Any] | None = None,
        filter: dict[str, Any] | None = None,
        select: int | slice | str | tuple[int | slice | str, ...] | None = None,
        target: str | Body | None = None, lightsource: LightSource | None = None,
        path: str | Path | None = None, frame: str | Frame | None = None,
        cadence: Cadence | None = None, fov: FOV | None = None,
        calibrations: Calibration | list[Calibration] | None = None,
        timeshift: tuple[float, str] | None = None,
        navigation: tuple[float, ...] | None = None, tracker: str | None = None,
        parallel: Observation | None = None,
        **kwargs: Any) -> list[Observation | ObsArray | None]: ...

def get_fits_image_hdu(hdulist: HDUList,
    index: int | str | None = None) -> PrimaryHDU | ImageHDU: ...
def read_pds3_image_array(label: Pds3Label, index: int | str = 0) -> ndarray: ...

##########################################################################################
