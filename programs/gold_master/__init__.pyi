##########################################################################################
# programs/gold_master/__init__.pyi
##########################################################################################
"""Type stub for :mod:`programs.gold_master`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any
from filecache import FCPath as FCPath, FileCache
from oops import Observation as Observation, Path as Path
from collections.abc import Callable
from numpy import ndarray, number
from polymath import Qube
from collections import defaultdict

# Parameters documented as a polymath type are passed through `as_scalar` and its
# siblings, so each accepts the class, a number, or a nested sequence of numbers.
# `str` is excluded deliberately: no polymath constructor accepts one.
_Numeric = float | number | list['_Numeric'] | tuple['_Numeric', ...]
QubeLike = Qube | ndarray | _Numeric

__all__ = ['set_default_obs', 'define_standard_obs', 'set_default_args', 'override',
           'module_dirname', 'set_gold_master_path', 'execute_as_command',
           'execute_as_pytest', 'run_tests', 'register_test_suite', 'get_test_suite',
           'BackplaneTest']

def set_default_obs(obspath: Path | str, index: int | tuple[int] | None,
    planets: str | list[str] | list, moons: str | list[str] | list | tuple = (),
    rings: str | list[str] | list | tuple = (), kwargs: dict | None = None) -> None: ...

def define_standard_obs(obsname: str, obspath: Path | str,
    index: int | tuple[int] | None = None, *,
    planets: str | list[str] | list | tuple = (),
    moons: str | list[str] | list | tuple = (),
    rings: str | list[str] | list | tuple = (), kwargs: dict | None = None) -> None: ...

def set_default_args(**options: Any) -> None: ...

def override(title: str, value: float | None,
    names: str | list[str] | list | None = None) -> None: ...

def module_dirname(module: str) -> str: ...

def set_gold_master_path(path: str) -> None: ...

def execute_as_command() -> None: ...

def execute_as_pytest(obsname: str = 'default') -> None: ...

def run_tests(args: Any) -> None: ...

def register_test_suite(name: str, func: Callable) -> None: ...

def get_test_suite(name: str) -> Callable: ...

class BackplaneTest:
    obs: Any
    overrides: Any
    args: Any
    suffix: Any
    upward: Any
    full_shape: Any
    task: Any
    derivs: Any
    undersample: Any
    inventory: Any
    border: Any
    body_names: Any
    limb_names: Any
    ring_names: Any
    ansa_names: Any
    planet_moon_pairs: Any
    planet_ring_pairs: Any
    origins: Any
    duv: Any
    meshgrids: Any
    backplanes: Any
    meshgrid: Any
    backplane: Any
    abspath: Any
    gold_dir: Any
    gold_arrays: Any
    gold_browse: Any
    output_dir: Any
    output_arrays: Any
    output_browse: Any
    sampled_gold: Any
    gold_summary_: Any
    summary: Any
    results: Any
    header: Any
    print_header: Any
    def __init__(self, obs: Observation, planets: list, moons: list, rings: list,
        overrides: dict, args: Any, suffix: str = '') -> None: ...
    def run_tests(self) -> None: ...
    def compare(self, array: QubeLike, master: QubeLike | float, title: str,
        limit: float | QubeLike = 0.0, method: str = '', operator: str = '=',
        radius: float = 0.0, mask: bool | ndarray = False) -> None: ...
    def gmtest(self, array: QubeLike, title: str, limit: float | QubeLike = 0.0,
        method: str = '', operator: str = '=', radius: float = 0.0,
        mask: bool | ndarray = False) -> None: ...
    def save_browse(self, array: QubeLike | ndarray, browse_path: FCPath) -> None: ...
    @staticmethod
    def read_browse(browse_path: FCPath) -> ndarray: ...
    @property
    def gold_summary(self) -> dict: ...
    def write_summary(self, outdir: FCPath) -> FCPath: ...

BACKPLANE_OUTPUT_PREFIX: FCPath

DEFAULTS: dict[str, Any]

GOLD_MASTER_PREFIX: FCPath | None

OOPS_BACKPLANE_OUTPUT_PATH: str

OOPS_GOLD_MASTER_PATH: str | None

STANDARD_OBS_INFO: dict[str, Any]

TEST_DATA_FILECACHE: FileCache

TEST_DATA_PREFIX: FCPath | None

TEST_OVERRIDES: defaultdict[str, dict[str, Any]]

TEST_SUITES: dict[str, Any]

##########################################################################################
