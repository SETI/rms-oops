# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`rms-oops` (imported as `oops`) is an observation-geometry library for planetary
science: it models instruments, spacecraft trajectories, and target bodies, and
computes per-pixel geometry ("backplanes") for real observations. It is built on
SPICE (via `cspyce`) and on the `polymath` array types.

Two library packages live under `src/`: `src/oops` and `src/spicedb`. Neither
contains test code; the tests for both live under `tests/`. The gold master
backplane test framework is `programs/gold_master`, imported as
`programs.gold_master`; it is a runnable tool rather than part of the `oops` API,
so it sits outside `src/` and outside the library. Its
`execute_as_pytest(obsname)` is what a host test calls to compare one standard
observation against its gold masters, and its `test_support` module resolves every
`$OOPS_*` resource path into a `filecache` prefix (`TEST_DATA_PREFIX`,
`TEST_SPICE_PREFIX`, `GOLD_MASTER_PREFIX`, `BACKPLANE_OUTPUT_PREFIX`) for both the
tool and the tests. Nothing under `src/` imports it, which is what keeps the test
scaffolding out of the published wheel.

## Environment

Nothing runs without external data. `OOPS_RESOURCES` must point at the resource
tree (SPICE kernels, test data, gold masters); the other variables derive from it.
The full list is in @README.md — read it rather than guessing paths. If tests or
imports fail because these are unset, say so; do not report it as a code defect.

## Build, test, lint

- Because the packages live under `src/`, nothing imports from a bare checkout.
  Work inside the virtualenv `./scripts/setup-venv.sh` creates, which installs
  `-e ".[dev]"`. Never install into system Python.
- `scripts/run-all-checks.sh` runs the gates; `-h` lists the flags. A full run is
  ruff, flake8, mypy, stubtest, pyroma, bandit, vulture, the three pytest suites,
  and the Sphinx build. The Sphinx gate builds the tree twice: the public copy into
  `docs/_build/html`, and a second copy tagged `private` (`-t private`) into
  `docs/_build/private/html`, in which autodoc documents the private members and the
  `.. private-only::` blocks of the Developer's Guide are rendered; `docs/conf.py`
  defines the tag's effects and that directive. The narrative pages are the User's
  Guide under `docs/user_guide/` and the Developer's Guide under `docs/dev_guide/`,
  one chapter per file; a change to the public API, a backplane, or a command-line
  option updates the matching chapter in the same change. pip-audit and PyMarkdown remain off by default (`ENABLE_*` in the
  script): pip-audit reports findings against pinned upstream dependencies this
  repository does not control, and PyMarkdown reports pre-existing findings in the
  Markdown. `.github/workflows/run-lint.yml` runs the same set minus the pytest
  suites, which belong to `run-tests.yml`; each gate is its own step so a failure
  names itself. Keep the two in step: the script is authoritative.
- Tests use **pytest**: module-level `test_*` functions and plain `assert`. There
  are no `unittest.TestCase` classes and no `runTest` methods.
  - `pytest tests --ignore=tests/hosts --ignore=tests/spicedb` — main suite
  - `pytest tests/hosts` — host tests, which are the gold masters
  - `pytest tests/spicedb` — the spicedb tests
  - `pytest tests` — all three; they pass together, but the check script and CI
    run them as three invocations so a failure is attributable to one suite.
- Shared fixtures live in `tests/conftest.py` (`core_kernels`, which furnishes the
  leap-second, PCK, and DE421 kernels with the Path and Frame registries cleared)
  and `tests/spicedb/conftest.py`. Setup that is specific to one module stays in
  that module as an `autouse` fixture.
- `-n auto` and `filterwarnings = ["error"]` are deliberately absent from
  `[tool.pytest.ini_options]` — read the comments there before adding either.
- Only files named `test_*.py` are collected. Several host modules
  (`tests/hosts/juno/**`, `tests/hosts/hst/__init__.py`,
  `tests/hosts/cassini/{uvis,vims}/*.py`) hold converted tests that no suite ran
  before the conversion either; they keep names pytest does not collect, and
  wiring them up means fixing the tests, not renaming the files.
- Gold-master tests for a single instrument take command-line options:
  `PYTHONPATH=. python tests/hosts/cassini/iss/gold_master.py --help`
- CI runs `scripts/automated_tests/oops_main_test.sh`, which reinstalls all
  dependencies from `-e ".[dev]"` and requires `SPICE_PATH`,
  `SPICE_SQLITE_DB_NAME`, and `OOPS_RESOURCES`. Keep it in step with
  `run-all-checks.sh`.
- **Ruff is the linter of record**, over the whole repository, and every check runs
  there except one. Ruff implements no rule in the E121-E133 range, so the
  continuation-line indent checks come from flake8 and nothing else: the gate is
  `flake8 --select=E12,E13`, the same split rms-polymath uses. `.flake8` configures
  that range and nothing else — it selects E12/E13 itself, so a bare `flake8`
  reports exactly what the gate reports, and it ignores the codes in that range the
  column-aligned house style would trip (leaving E123, E125 and E133 enforced).
  Every other ignore lives in `[tool.ruff.lint]`; do not add non-continuation codes
  to `.flake8`, because nothing there is ever selected.
- `ruff format` is deliberately never run. Column-aligned assignments, imports,
  and trailing comments are the house style and the formatter would undo them.
- mypy covers `tests/` only. Under `src`, the only inline annotations are the
  return types of properties and setters (plus the signatures of `fittable.py`,
  `mutable.py`, `frame_.py` and `event.py`); method parameter and return types
  live in the docstrings, so checking `src` would report their absence rather
  than any defect. `mypy_path` does not name `src`, so that run does not see the
  stubs either; pointing it there reports the tests' use of private members,
  which the stubs do not publish.
- All three packages ship a PEP 561 `py.typed` marker, one per top-level package
  (`src/oops`, `src/spicedb`, `programs`); a marker at the package root covers the
  whole tree, so subpackages do not carry their own. The published type information
  lives in one `.pyi` per directory, `__init__.pyi`, and nowhere else. A name is
  therefore annotated when it is imported from the package that exports it, as in
  `from oops.frame import SpiceFrame`, and not when it is imported from the module
  that defines it: the latter falls back to the unannotated source. Each package
  stub declares its classes outright rather than re-exporting them, because a
  re-export would resolve back to that source.
- A stub replaces its module outright for a type checker, so whatever it omits is
  invisible downstream and adding, renaming or re-signing any public member means
  updating `__init__.pyi` in the same change. `stubtest` enforces this and runs in
  the check script and in CI with `--ignore-missing-stub`, since the modules below
  each package no longer have stubs of their own. The members bound on at import
  time need writing out by hand: the ~86 backplane methods
  `_define_backplane_names` attaches all have to appear in
  `backplane/__init__.pyi`, as do the photon-solver methods on `Path` and
  `Surface` and the class constants (`Frame.J2000`, `Path.SSB`,
  `Transform.IDENTITY`, the `Gravity` bodies) that their modules assign after the
  class statement.
- Types come from the docstrings, and the two must agree: a parameter documented as
  `ScalarLike` is annotated `ScalarLike` in the stub, and changing one means changing
  the other. A parameter documented as a polymath type takes the matching
  `ScalarLike`, `PairLike`, `Vector3Like` alias, because the body passes it through
  `as_scalar` or a sibling and so accepts the class, a number, or a nested sequence of
  numbers; a return takes the exact class. `Any` remains only where the docstring gives
  no type. The aliases are `polymath`'s, imported from `polymath.typedefs`; the stubs
  do not define their own. `ScalarLike`, `BooleanLike`, `QubeLike` and `PairLike` each
  admit a lone number, the last because `Pair.as_pair(1)` repeats the value; the other
  shape-constrained aliases (`Vector3Like`, `VectorLike`, `MatrixLike`, `Matrix3Like`,
  `QuaternionLike`) do not, so a parameter that broadcasts one needs the number spelled
  out alongside. `stubtest-allowlist.txt` holds the names that exist only at run time,
  chiefly the cross-class attributes injected to break circular imports.
- The legacy exclusions are recorded in `pyproject.toml`: `ideas/` and the parked,
  uncollected test modules are outside ruff's scope, and `src/oops/hosts/*` and
  `src/spicedb/*` carry per-file ignores.

## Code style

`.flake8` ignores E501 and most whitespace/continuation checks. Those ignores are
deliberate: column-aligned assignments, imports, and trailing comments are the
house style. Do not "fix" alignment or blank-line counts that flake8 passes.

- Line length is **90 columns everywhere**. One line exceeds it,
  `src/oops/hosts/jwst/__init__.py:33`, which is a bare URL that cannot be
  wrapped without breaking it.
- Every file opens with a banner of exactly 90 `#` characters, then
  `# oops/path/to/file.py: description`, then the banner again; the file's last
  line is a closing banner of the same width. Every horizontal rule made of `#`
  is 90 columns wide, indent included. A lone `#` on its own line is a blank
  line inside a comment paragraph, not a rule, and stays as it is. The banner
  names the path from the import root, so a module under `src/oops` says
  `oops/...` with no `src/` prefix, and one under `programs` says `programs/...`.
- There are no `#===...` or `#---...` separator rules. They used to sit above
  each `def` in the legacy modules; do not reintroduce them.
- Docstrings come in two coexisting styles. Modern: Google-ish, using
  `Parameters:` (never `Args:`), a class-level `Attributes:` block, and
  noun-phrase summary lines. Legacy: two-column `Input:` / `Return:` blocks
  hanging at column 25. Match the file.
- A subpackage directory shares its name with the abstract class it exports, so
  the defining module gets a trailing underscore to avoid shadowing the package:
  `frame/frame_.py` defines `Frame`, `path/path_.py` defines `Path`, likewise
  `surface_`, `fov_`, `observation_`, `cadence_`, `calibration_`, `gravity_`.
  Subclass modules are lowercase with no separators (`twovectorframe.py`).
- Imports are absolute; `polymath` is grouped with the `oops` imports, not treated
  as third-party.
- Every property getter is annotated `-> T` and every setter `-> None`; a name
  not yet bound when the class body runs is quoted, and a class that cannot be
  imported at run time without a cycle is imported under `TYPE_CHECKING`. Ordinary
  methods and functions carry no signature annotations: every parameter and return
  is typed in the docstring instead, `name (type): ...` under `Parameters:` and
  `type: ...` under `Returns:`, with `A | B` for alternatives (never `A or B`),
  `None` last, and containers parameterized where the content is known. A
  method, function, class or attribute named in docstring prose takes a Sphinx
  role (`:meth:`, `:func:`, `:class:`, `:attr:`) so that Napoleon links it;
  bare backticks are for parameter names.
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
- Dependencies and tool configuration live in `pyproject.toml` only.
  `requirements.txt` contains just `-e .`, and there is no `setup.cfg` or
  `.coveragerc`. Versions come from `setuptools_scm`; never hand-edit
  `src/oops/_version.py`.
- `.claude/rules/` and `.claude/skills/` are generic RMS-wide templates and are
  still partly aspirational here. Where they conflict with this repository —
  Ruff vs. flake8, a Sphinx docs tree — **the repository's actual conventions
  win.**
