Repository layout
=================

Two library packages live under ``src``, and nothing else is importable from a bare
checkout: work inside the virtual environment that ``scripts/setup-venv.sh`` creates,
which installs the packages in editable mode. The tests live under ``tests`` and the
gold master tool under ``programs``, outside the library and outside the wheel.

::

    rms-oops/
    ├── src/
    │   ├── oops/                  The library. Importable as `oops`; ships in the wheel.
    │   │   ├── __init__.py        Imports every subpackage, then injects the cross-class
    │   │   │                      attributes that break the circular imports.
    │   │   ├── __init__.pyi       Type stub for the top-level namespace.
    │   │   ├── py.typed           PEP 561 marker; covers the whole package.
    │   │   ├── backplane/         Backplane, and one module per family of backplanes.
    │   │   ├── body.py            Body: a named target and its path, frame, surface,
    │   │   │                      gravity and rings; define_solar_system().
    │   │   ├── cadence/           Cadence and its subclasses: the timing of an exposure.
    │   │   ├── calibration/       Calibration and its subclasses: data numbers to units.
    │   │   ├── config.py          The run-time switches: QUICK, LOGGING, PATH_PHOTONS...
    │   │   ├── constants.py       C, AU, DPR and the other numeric constants.
    │   │   ├── event.py           Event: a photon at a point in spacetime.
    │   │   ├── fittable.py        Fittable: the interface of an object with parameters.
    │   │   ├── fov/               FOV and its subclasses: pixel coordinates to lines of
    │   │   │                      sight.
    │   │   ├── frame/             Frame and its subclasses; Transform lives alongside.
    │   │   ├── gravity/           Gravity and OblateGravity: orbital frequencies.
    │   │   ├── hosts/             One package per mission and instrument; each exports
    │   │   │                      from_file(). Not covered by this guide.
    │   │   ├── lightsource/       LightSource and DiskSource: the Sun and the stars.
    │   │   ├── meshgrid.py        Meshgrid: the pixel grid a Backplane evaluates over.
    │   │   ├── mutable.py         Mutable: the refresh/freeze protocol for objects that
    │   │   │                      may contain a Fittable.
    │   │   ├── observation/       Observation and its subclasses: Snapshot, Pushbroom...
    │   │   ├── oops.py            Oops: the empty common ancestor.
    │   │   ├── path/              Path and its subclasses; the path photon solver.
    │   │   ├── spice_support.py   Furnishing kernels and translating SPICE identifiers.
    │   │   ├── surface/           Surface and its subclasses; the surface photon solver.
    │   │   ├── transform.py       Transform: a rotation between two frames at a time.
    │   │   └── _cache.py          _Cache: the bounded cache that KeplerPath and several
    │   │                          frames use.
    │   └── spicedb/               Kernel selection from a SQLite database. Ships in the
    │                              wheel; documented by its API reference only.
    ├── programs/
    │   ├── gold_master/           The gold master backplane test framework, imported as
    │   │                          `programs.gold_master`. A runnable tool, not library
    │   │                          code; nothing under src imports it.
    │   └── py.typed
    ├── tests/                     The pytest suites, mirroring src/oops.
    │   ├── conftest.py            Shared fixtures and the --gold-master option.
    │   ├── backplane/ ... surface/  One directory per subpackage of oops.
    │   ├── hosts/                 The gold master tests, one package per instrument.
    │   ├── programs/              Unit tests of the gold master framework itself.
    │   └── spicedb/               The spicedb tests, with their own conftest.py.
    ├── docs/                      The Sphinx tree: conf.py, the API pages, and the two
    │                              guides under user_guide/ and dev_guide/.
    ├── scripts/
    │   ├── setup-venv.sh          Creates venv/ and installs `-e ".[dev]"`.
    │   ├── run-all-checks.sh      Runs every quality gate; authoritative for CI.
    │   └── automated_tests/       The script the test workflow runs.
    ├── .github/workflows/         run-lint.yml, run-tests.yml, run-windows-tests.yml
    │                              and the two PyPI publishing workflows.
    ├── ideas/                     Scratch and deprecated drafts. Not importable, not
    │                              linted, not part of either package.
    ├── critiques/                 Review reports on the source.
    ├── pyproject.toml             Packaging, dependencies, and every tool's configuration.
    ├── .flake8                    The one setting ruff cannot hold: the continuation-line
    │                              checks.
    ├── stubtest-allowlist.txt     Names that exist only at run time.
    ├── CLAUDE.md                  The working notes on conventions and traps.
    └── README.md                  Setup and the environment variables.

The importable public packages are ``oops`` and ``spicedb``. Everything else supports
them: ``programs.gold_master`` is importable only with the repository root on the path,
which is how the tests and the check script run it.

Naming inside a subpackage
--------------------------

Each subpackage directory shares its name with the abstract class it exports, so the
module defining that class takes a trailing underscore to avoid shadowing the package:
``frame/frame_.py`` defines :class:`~oops.Frame`, ``path/path_.py`` defines
:class:`~oops.Path`, and likewise ``surface_``, ``fov_``, ``observation_``, ``cadence_``,
``calibration_`` and ``gravity_``. Subclass modules are lowercase with no separators, one
class per module: ``twovectorframe.py``, ``graphicellipsoid.py``, ``linearcoordpath.py``.

Each subpackage's ``__init__.py`` imports its classes and lists them in ``__all__``; the
companion ``__init__.pyi`` is the only type stub for that subpackage and declares the same
names. :doc:`dev_guide_extending` says what to update in each when a class is added.
