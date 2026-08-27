# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`rms-oops` (imported as `oops`) is an observation-geometry library for planetary
science: it models instruments, spacecraft trajectories, and target bodies, and
computes per-pixel geometry ("backplanes") for real observations. It is built on
SPICE (via `cspyce`) and on the `polymath` array types.

## Environment

Nothing runs without external data. `OOPS_RESOURCES` must point at the resource
tree (SPICE kernels, test data, gold masters); the other variables derive from it.
The full list is in @README.md — read it rather than guessing paths. If tests or
imports fail because these are unset, say so; do not report it as a code defect.

## Build, test, lint

- Tests use `unittest`, **not** pytest. There is no `conftest.py` and no pytest
  config. Test classes define a single `runTest` method.
  - `python -m unittest tests/unittester.py` — main suite
  - `python -m unittest tests/unittester_with_hosts.py` — main suite plus hosts
  - `python -m unittest tests/hosts/unittester.py` — host tests incl. gold masters
- Each test subdirectory has its own `unittester.py` that aggregates its modules.
  A new test module is not run until it is added to the enclosing `unittester.py`.
- Gold-master tests for a single instrument take command-line options:
  `PYTHONPATH=. python tests/hosts/cassini/iss/gold_master.py --help`
- CI runs `scripts/automated_tests/oops_main_test.sh`, which reinstalls all
  dependencies from `requirements.txt` and requires `SPICE_PATH`,
  `SPICE_SQLITE_DB_NAME`, and `OOPS_RESOURCES`.
- Lint is flake8, on two targets: `flake8 oops` and `flake8 spicedb`.

## Code style

`.flake8` ignores E501 and most whitespace/continuation checks. Those ignores are
deliberate: column-aligned assignments, imports, and trailing comments are the
house style. Do not "fix" alignment or blank-line counts that flake8 passes.

- Line length is **90 columns in refactored modules** (`oops/frame/`, `oops/path/`,
  `cache.py`, `fittable.py`, `mutable.py`) and **80 in legacy modules**. Match the
  file you are editing.
- Every file opens with a banner of `#` characters at the file's line width, then
  `# oops/path/to/file.py: description`, then the banner again; the file's last
  line is a closing banner of the same width. Legacy files also put a
  `#===...` separator above each `def`. Preserve all of these.
- Docstrings come in two coexisting styles. Modern: Google-ish, using
  `Parameters:` (never `Args:`), with class-level `Properties:` bullets and
  noun-phrase summary lines. Legacy: two-column `Input:` / `Return:` blocks
  hanging at column 25. Match the file.
- A subpackage directory shares its name with the abstract class it exports, so
  the defining module gets a trailing underscore to avoid shadowing the package:
  `frame/frame_.py` defines `Frame`, `path/path_.py` defines `Path`, likewise
  `surface_`, `fov_`, `observation_`, `cadence_`, `calibration_`, `gravity_`.
  Subclass modules are lowercase with no separators (`twovectorframe.py`).
- Imports are absolute; `polymath` is grouped with the `oops` imports, not treated
  as third-party. Type annotations appear only in `fittable.py` and `mutable.py`.
- Every `__init__.py` declares its public API in `__all__`: the re-exported names
  for a package that only re-exports, the public classes and functions it defines
  otherwise. A module that exists only for its import side effects (`all.py`)
  carries `# flake8: noqa: F401` instead.

## Architecture traps

- **Always `import oops` first.** The bottom of `oops/__init__.py` injects
  cross-class attributes (`Transform._Frame`, `Frame._Path`, `Event.SSB`, …) to
  break circular imports. Importing a leaf module on its own leaves those `None`
  and fails obscurely far from the cause.
- Geometry values are `polymath` types (`Scalar`, `Vector3`, `Qube`), not numpy
  arrays. Data under a mask is garbage — combine with `.antimask` before using
  `.vals`.
- Units are implicit and bare: **km, seconds TDB, radians**. `polymath.Units` is
  effectively unused, so a returned value is not self-describing and unit-tagged
  objects are not honored. Angles are radians even where a label says "deg".
- `Event` objects and registered backplane arrays are `as_readonly()` and shared
  through caches. Mutating one in place corrupts every other user of it.
- Path and Frame registries and their registration hook are private
  (`_PATH_REGISTRY`, `_FRAME_REGISTRY`, `_register()`).
- Backplane methods are attached by `Backplane._define_backplane_names(globals().copy())`
  at the end of each `oops/backplane/*.py`. That sweeps up *every* module-level
  function, imported helpers included — don't leave stray functions in those modules.
- `quick=True` **disables** QuickPath/QuickFrame optimization. `for_path`/`for_frame`
  accept only `None` or a dict; anything else silently returns the unoptimized object.

## Repo conventions

- All changes go through a pull request into `main`.
- Python 3.11 is the minimum supported version; see `requires-python` in
  `pyproject.toml` and the CI matrices.
- `.claude/rules/` and `.claude/skills/` are generic RMS-wide templates and are
  aspirational here. Where they conflict with this repository — Ruff vs. flake8,
  a `src/` layout, a Sphinx docs tree, `scripts/run-all-checks.sh` —
  **the repository's actual conventions win.**
