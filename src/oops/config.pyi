##########################################################################################
# oops/config.pyi
##########################################################################################
"""Type stub for :mod:`oops.config`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. The stub describes the shape of the API exactly: every public
name, its parameters, which of them are keyword-only, and which have defaults. Types are
given where they are unambiguous and are `Any` elsewhere.
"""

from typing import Any

class QUICK:
    flag: bool
    dictionary: Any
class PATH_PHOTONS:
    max_iterations: int
    dlt_precision: float
    dlt_limit: float
    km_precision: float
    rel_precision: float
class SURFACE_PHOTONS:
    max_iterations: int
    dlt_precision: float
    dlt_limit: float
    collapse_threshold: float
    km_precision: float
    rel_precision: float
class EVENT_CONFIG:
    collapse_threshold: float
LOG_DATEFMT: str
LOG_FORMAT: str
LOG_FORMATTER: Any
LOGGING_STACK: Any

class LOGGING:
    prefix: str
    quickpath_creation: bool
    quickframe_creation: bool
    fov_iterations: bool
    path_iterations: bool
    surface_iterations: bool
    observation_iterations: bool
    event_time_collapse: bool
    surface_time_collapse: bool
    stdout: bool
    stderr: bool
    file_path: Any
    logger: Any
    level: Any
    handlers: Any
    log_formatting: bool
    warnings: int
    errors: int
    lines: int
    python_warnings: bool
    LEVELS: dict[str, int]
    LEVEL_NAMES: dict[int, str]
    @staticmethod
    def reset() -> None: ...
    @staticmethod
    def all(flag: Any, category: str = '', reset: bool = False) -> None: ...
    @staticmethod
    def off(category: str = '', reset: bool = True) -> None: ...
    @staticmethod
    def on(prefix: str = '   ', category: str = '', reset: bool = False) -> None: ...
    @staticmethod
    def set_stdout(flag: Any, reset: bool = False) -> None: ...
    @staticmethod
    def set_stderr(flag: Any, reset: bool = False) -> None: ...
    @staticmethod
    def set_file(file_path: str = '', reset: bool = False) -> None: ...
    @staticmethod
    def set_logger(logger: Any = None, level: str = 'DEBUG',
        reset: bool = False) -> None: ...
    @staticmethod
    def set_logger_level(level: Any) -> None: ...
    @staticmethod
    def print(*args: Any, level: Any = ..., literal: bool = False,
        force: bool = False) -> None: ...
    @staticmethod
    def debug(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def info(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def warn(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def error(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def fatal(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def convergence(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def diagnostic(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def diagnostics(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def performance(*args: Any, force: bool = False) -> None: ...
    @staticmethod
    def exception(exception: Any, message: str = '') -> None: ...
    @staticmethod
    def literal(*args: Any, level: Any = ..., force: bool = True) -> None: ...
    @staticmethod
    def push() -> None: ...
    @staticmethod
    def pop() -> None: ...
class PICKLE_CONFIG:
    quickpath_details: bool
    quickframe_details: bool
    backplane_events: bool
class AREA_FACTOR:
    old: bool

##########################################################################################
