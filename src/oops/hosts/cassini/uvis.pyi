##########################################################################################
# oops/hosts/cassini/uvis.pyi
##########################################################################################
"""Type stub for :mod:`oops.hosts.cassini.uvis`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

from oops.hosts.cassini import Cassini as Cassini

DEBUG: bool

def from_file(filespec: Any, data: bool = True, enclose: bool = False,
    method: str = 'strict', **parameters: Any) -> Any: ...

def get_qube(filespec: Any, tstart: Any, label: Any, data: Any, enclose: Any) -> Any: ...

def get_one_qube(label: Any, detector: Any, resolution: Any, fov: Any, cadence: Any,
    frame_id: Any, shape: Any, array: Any, samples: Any, lines: Any, line0: Any,
    line1: Any, line_bin: Any, bands: Any, band0: Any, band1: Any, band_bin: Any,
    rebin: Any) -> Any: ...

def get_time_series(filespec: Any, tstart: Any, label: Any, data: Any) -> Any: ...

def get_spectrum(filespec: Any, tstart: Any, label: Any, data: Any) -> Any: ...

def load_data(filespec: Any, body: Any, dtype: Any) -> Any: ...

def initialize(ck: str = 'reconstructed', planets: Any = None, asof: Any = None,
    spk: str = 'reconstructed', gapfill: bool = True, mst_pck: bool = True,
    irregulars: bool = True) -> None: ...

class UVIS:
    instrument_kernel: Any
    fovs: Any
    initialized: bool
    abbrevs: Any
    frame_ids: Any
    @staticmethod
    def initialize(ck: str = 'reconstructed', planets: Any = None, asof: Any = None,
        spk: str = 'reconstructed', gapfill: bool = True, mst_pck: bool = True,
        irregulars: bool = True) -> None: ...
    @staticmethod
    def reset() -> None: ...

##########################################################################################
