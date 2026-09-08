##########################################################################################
# oops/hosts/keck/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.keck`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.
"""

import pathlib
from typing import Any
from astropy.io.fits import HDUList
from filecache import FCPath as FCPath
from numpy import ndarray
from oops import Body, Observation
from oops.observation import Snapshot

__all__ = ['from_file', 'Keck']

def from_file(filespec: str | pathlib.Path | FCPath,
    **parameters: Any) -> Observation: ...

class Keck:
    def filespec(self, keck_file: HDUList) -> str: ...
    def telescope_name(self, keck_file: HDUList) -> str: ...
    def instrument_name(self, keck_file: HDUList) -> str: ...
    def detector_name(self, keck_file: HDUList, **parameters: Any) -> str: ...
    def data_array(self, keck_file: HDUList, **parameters: Any) -> ndarray: ...
    def time_limits(self, keck_file: HDUList,
        **parameters: Any) -> tuple[float, float]: ...
    def target_body(self, keck_file: HDUList) -> Body: ...
    def construct_snapshot(self, keck_file: HDUList, **parameters: Any) -> Snapshot: ...
    @staticmethod
    def from_opened_fitsfile(keck_file: HDUList, **parameters: Any) -> Observation: ...

##########################################################################################
