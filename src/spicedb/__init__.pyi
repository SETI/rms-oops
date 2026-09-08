##########################################################################################
# spicedb/__init__.pyi
##########################################################################################
"""Type stub for :mod:`spicedb`.

The source types its methods in their docstrings rather than in their signatures, so the
type information for public symbols is published here instead. Only package stubs exist,
so a name is annotated when it is imported from the package that exports it and not when
it is imported from the module that defines it. The stub describes the shape of the API
exactly: every public name, its parameters, which of them are keyword-only, and which have
defaults. Types are given where they are unambiguous and are `Any` elsewhere.
"""

from collections.abc import Callable
from typing import Any
from filecache import FCPath as FCPath, FileCache as FileCache

__all__ = ['lrange', 'KernelInfo', 'kernels_from_filespec', 'set_spice_path',
           'get_spice_path', 'get_spice_filecache', 'get_spice_filecache_prefix',
           'open_db', 'close_db', 'db_is_open', 'set_translator', 'select_lsk',
           'select_pck', 'select_spk', 'select_inst', 'select_ck', 'select_by_name',
           'select_by_filespec', 'as_dict', 'furnish_kernels', 'furnish_lsk',
           'furnish_pck', 'furnish_spk', 'furnish_inst', 'furnish_ck', 'furnish_by_name',
           'furnish_by_metafile', 'furnish_by_filepath', 'unload_by_name',
           'unload_by_type', 'unload_by_filepath', 'unload_all', 'as_names',
           'furnished_names', 'furnished_basenames', 'used_basenames',
           'furnish_cassini_kernels', 'furnish_solar_system']

def lrange(*args: int) -> list[int]: ...

class KernelInfo:
    kernel_name: str
    kernel_version: str | None
    kernel_type: str
    filespec: str
    start_time: str | None
    stop_time: str | None
    release_date: str | None
    spice_id: int | None
    load_priority: int
    basename: str
    start_tai: float
    stop_tai: float
    start_tdb: float
    stop_tdb: float
    file_no: int | None
    def __init__(self, info: list | tuple) -> None: ...
    def compare(self, other: KernelInfo) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __le__(self, other: KernelInfo) -> bool: ...
    def __lt__(self, other: KernelInfo) -> bool: ...
    def __ge__(self, other: KernelInfo) -> bool: ...
    def __gt__(self, other: KernelInfo) -> bool: ...
    @property
    def full_name(self) -> str: ...
    @property
    def timeless(self) -> bool: ...

def kernels_from_filespec(filespec: str, name: str | None = None,
    version: str | None = None, release: str | None = None,
    priority: int = 100) -> list[KernelInfo]: ...

def set_spice_path(spice_path: str | FCPath = '') -> None: ...

def get_spice_path() -> str | FCPath: ...

def get_spice_filecache() -> FileCache: ...

def get_spice_filecache_prefix() -> FCPath: ...

def open_db(name: FCPath | None = None) -> None: ...

def close_db() -> None: ...

def db_is_open() -> bool: ...

def set_translator(func: Callable[[str], str | None]) -> None: ...

def select_lsk(asof: str | None = None, after: str | None = None,
    redo: bool = True) -> list[KernelInfo]: ...

def select_pck(bodies: int | list | tuple | None = None, name: str | None = None,
    asof: str | None = None, after: str | None = None,
    redo: bool = True) -> list[KernelInfo]: ...

def select_spk(bodies: int | list | tuple, name: str | None = None,
    time: tuple | None = None, asof: str | None = None, after: str | None = None,
    redo: bool = True) -> list[KernelInfo]: ...

def select_inst(ids: int | list | tuple, inst: str | list | tuple | None = None,
    types: str | list | tuple | None = None, asof: str | None = None,
    after: str | None = None, redo: bool = True) -> list[KernelInfo]: ...

def select_ck(ids: int | list | tuple, name: str | None = None, time: tuple | None = None,
    asof: str | None = None, after: str | None = None,
    redo: bool = True) -> list[KernelInfo]: ...

def select_by_name(names: str | list[str],
    time: tuple | None = None) -> list[KernelInfo]: ...

def select_by_filespec(filespecs: str | list[str],
    time: tuple | None = None) -> list[KernelInfo]: ...

def as_dict(kernel_list: list[KernelInfo]) -> dict[str, Any]: ...

def furnish_kernels(kernel_list: list[KernelInfo], fast: bool = True) -> list[str]: ...

def furnish_lsk(asof: str | None = None, after: str | None = None, redo: bool = True,
    fast: bool = True) -> list[str]: ...

def furnish_pck(bodies: int | list | tuple | None = None, name: str | None = None,
    asof: str | None = None, after: str | None = None, redo: bool = True,
    fast: bool = True) -> list[str]: ...

def furnish_spk(bodies: int | list | tuple, name: str | None = None,
    time: tuple | None = None, asof: str | None = None, after: str | None = None,
    redo: bool = True, fast: bool = True) -> list[str]: ...

def furnish_inst(ids: int | list | tuple, inst: str | list | tuple | None = None,
    types: str | list | tuple | None = None, asof: str | None = None,
    after: str | None = None, redo: bool = True, fast: bool = True) -> list[str]: ...

def furnish_ck(ids: int | list | tuple, name: str | None = None,
    time: tuple | None = None, asof: str | None = None, after: str | None = None,
    redo: bool = True, fast: bool = True) -> list[str]: ...

def furnish_by_name(names: str | list[str], time: tuple | None = None,
    fast: bool = True) -> list[str]: ...

def furnish_by_metafile(metafile: str | FCPath, time: tuple | None = None,
    asof: str | None = None) -> list[str]: ...

def furnish_by_filepath(filepath: str) -> None: ...

def unload_by_name(names: str | list[str]) -> None: ...

def unload_by_type(types: str | list[str] | None = None) -> None: ...

def unload_by_filepath(filepath: str) -> None: ...

def unload_all() -> None: ...

def as_names(kernels: list[KernelInfo]) -> list[str]: ...

def furnished_names(types: str | list[str] | None = None) -> list[str]: ...

def furnished_basenames(types: str | list[str] | None = None) -> list[str]: ...

def used_basenames(types: str | list[str] = [],
    time: str | float | tuple[str | float, str | float] | None = None,
    bodies: list[int] = [], sc: int | None = None, inst: str | None = None,
    slop: float = ...) -> list[str]: ...

def furnish_cassini_kernels(start_time: str, stop_time: str,
    instrument: list[str] | None = None, asof: str | None = None) -> list[str]: ...

def furnish_solar_system(start_time: str | float | None = None,
    stop_time: str | float | None = None, asof: str | None = None,
    planets: int | tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)) -> list[str]: ...

##########################################################################################
