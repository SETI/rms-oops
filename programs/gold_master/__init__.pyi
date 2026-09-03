##########################################################################################
# programs/gold_master/__init__.pyi
##########################################################################################
"""Type stub for :mod:`programs.gold_master`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

__all__ = ['set_default_obs', 'define_standard_obs', 'set_default_args', 'override',
           'module_dirname', 'set_gold_master_path', 'execute_as_command',
           'execute_as_pytest', 'run_tests', 'register_test_suite', 'get_test_suite',
           'BackplaneTest']

def module_dirname(module: Any) -> Any: ...

def set_gold_master_path(path: Any) -> None: ...

def set_default_obs(obspath: Any, index: Any, planets: Any, moons: Any = (),
    rings: Any = (), kwargs: Any = None) -> None: ...

def define_standard_obs(obsname: Any, obspath: Any, index: Any = None, *,
    planets: Any = (), moons: Any = (), rings: Any = (), kwargs: Any = None) -> None: ...

def set_default_args(**options: Any) -> None: ...

def override(title: Any, value: Any, names: Any = None) -> None: ...

def execute_as_command() -> None: ...

def execute_as_pytest(obsname: str = 'default') -> None: ...

def run_tests(args: Any) -> None: ...

def register_test_suite(name: Any, func: Any) -> None: ...

def get_test_suite(name: Any) -> Any: ...

class _BackplaneComparison:
    STATUS_LEVEL: Any
    title: str
    suite: str
    limit: float
    method: str
    operator: str
    radius: float
    mask: bool
    pickle_path: str
    status: str
    max_diff1: float
    diff_errors1: int
    mask_errors1: int
    distance: float
    max_diff2: float
    diff_errors2: int
    mask_errors2: int
    pixels: int
    def __init__(self, **kwargs: Any) -> None: ...
    @property
    def logging_level(self) -> Any: ...
    @staticmethod
    def set_no_gold_master_status(is_ok: bool = False) -> None: ...
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
    def __init__(self, obs: Any, planets: Any, moons: Any, rings: Any, overrides: Any,
        args: Any, suffix: str = '') -> None: ...
    def run_tests(self) -> None: ...
    def compare(self, array: Any, master: Any, title: Any, limit: float = 0.0,
        method: str = '', operator: str = '=', radius: float = 0.0,
        mask: bool = False) -> None: ...
    def gmtest(self, array: Any, title: Any, limit: float = 0.0, method: str = '',
        operator: str = '=', radius: float = 0.0, mask: bool = False) -> None: ...
    def save_browse(self, array: Any, browse_path: Any) -> None: ...
    @staticmethod
    def read_browse(browse_path: Any) -> Any: ...
    @property
    def gold_summary(self) -> Any: ...
    def write_summary(self, outdir: Any) -> Any: ...

##########################################################################################
