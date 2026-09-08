Testing and validation
======================

The unit tests
--------------

The tests under ``tests`` mirror ``src/oops`` directory for directory, and every module
is named ``test_<module>.py``. A test is a module-level function with a sentence-length
name, a one-line docstring stating the claim, type annotations including ``-> None``,
and plain ``assert`` statements; floating-point results are compared with
``pytest.approx`` or ``numpy.allclose``. From ``tests/backplane/test_distance.py``:

.. code-block:: python

    def test_distance_is_light_time_times_the_speed_of_light(bp: Backplane) -> None:
        """Distance and light time describe the same photon path."""

        distance = _unmasked(bp.distance(PLANET))
        light_time = _unmasked(bp.light_time(PLANET))

        assert np.allclose(distance, light_time * C)

Fixtures replace setup and teardown. ``tests/conftest.py`` provides ``core_kernels``,
which furnishes the leap-second, PCK and DE421 kernels and clears the Path and Frame
registries before and after the test, so that a path one module registers cannot leak
into another. A module that needs the registries empty but no kernels declares its own
``autouse`` fixture calling ``Path._reset_caches()`` and ``Frame._reset_caches()``.
``tests/backplane/conftest.py`` goes further: it defines a whole solar system once for
the package, builds a synthetic Saturn observation from a
:class:`~oops.frame.TwoVectorFrame` pointed at the planet and a
:class:`~oops.fov.FlatFOV`, and hands every backplane test one shared
:class:`~oops.Backplane` over it, with a function-scoped ``fresh_bp`` for tests that
mutate. It is package-scoped deliberately: an event cached by one backplane holds
references to the waypoints and wayframes current when it was built, and rebuilding the
registries between modules would leave every such event stale.

Write tests against the package exports (``from oops.frame import SpinFrame``) rather
than the defining module, so that the stubs type them and mypy can check the test.
Import order matters as everywhere: ``import oops`` or a subpackage of it comes first.

There is no ``-n auto`` and no ``filterwarnings = error``; the reasons are in the
comments of ``[tool.pytest.ini_options]``, and adding either means addressing them.

The gold master tests
---------------------

The unit tests check internal consistency on a synthetic observation. The gold master
tests check the real thing: for a standard observation of each instrument, the
framework in :mod:`programs.gold_master` computes some two hundred backplanes and
compares each, pixel by pixel, against a stored reference array, the *gold master*. A
change that alters any geometric result shows up here, with the size of the difference
and where it occurs. The :doc:`User's Guide </user_guide/user_guide_gold_master>`
documents the command line in full; this section is about using the framework as a
developer.

How a host test is wired
~~~~~~~~~~~~~~~~~~~~~~~~

Each instrument that has gold masters has a package under ``tests/hosts`` with three
files. ``standard_obs.py`` registers the observations and the module that reads them:

.. code-block:: python

    import programs.gold_master as gm

    gm.define_standard_obs('W1573721822_1',
            obspath = 'cassini/ISS/W1573721822_1.IMG',
            index   = None,
            planets = ['SATURN'],
            moons   = ['EPIMETHEUS'],
            rings   = ['SATURN_MAIN_RINGS'])

    gm.set_default_args(module='oops.hosts.cassini.iss')

    gm.override('SATURN longitude d/du self-check (deg/pix)', 0.3)

``obspath`` is relative to ``$OOPS_TEST_DATA_PATH``; ``index`` selects one observation
when the host's ``from_file`` returns a list; ``planets``, ``moons`` and ``rings`` are
the default targets, each a registered body name; ``set_default_args`` fixes the host
module and any option defaults for the instrument (Galileo SSI and JunoCam turn the
inventory off and widen the border, for instance); and ``override`` loosens or cancels
one named comparison, with ``None`` cancelling it. ``gold_master.py`` is the command-line
entry point, which imports ``standard_obs`` for its side effect and calls
:func:`~programs.gold_master.execute_as_command`. ``test_gold_master.py`` is the pytest
module, one test per observation calling
:func:`~programs.gold_master.execute_as_pytest` with its name, wrapped in an ``autouse``
fixture that undefines the solar system before and after, because SPICE gives a later
kernel precedence and a second observation in the same process would otherwise see the
first one's kernels.

What the framework does
~~~~~~~~~~~~~~~~~~~~~~~

For each observation, ``BackplaneTest`` reads the file through the host's ``from_file``,
builds the list of surface keys to test (each body, its ``:LIMB``, and for a body with
rings its ``:RING`` and ``:ANSA``; each named ring and its ``:ANSA``), and constructs five
backplanes: one on a meshgrid with its origin at the pixel centers, undersampled by the
``--undersample`` factor (16 by default), and four more offset by a small step in *u*
and *v* for the derivative tests. Every registered test suite then runs against it.

A suite calls two methods. ``gmtest(array, title, limit=0., method='', operator='=',
radius=0., mask=False)`` compares a backplane against the stored master of the same
title, and ``compare(array, master, title, ...)`` compares against a value the suite
supplies, which is how analytic identities are checked (a clock angle plus a position
angle is zero, a longitude measured two ways differs by 180 degrees). ``limit`` is the
tolerance, scaled by the command line's ``--tolerance`` factor; ``method`` is ``''``,
``'degrees'`` or ``'mod360'`` (compare in degrees, the latter modulo 360) or ``'border'``;
``operator`` allows an inequality; ``radius`` is the number of pixels by which a value or
a mask edge may be shifted and still pass, which absorbs the sub-pixel pointing jitter
between SPICE kernel versions; ``mask`` excludes pixels from the comparison. The
comparison machinery first checks the values and masks directly, then, within the
radius, whether each discrepant pixel is explained by the local gradient or by a value
found among its neighbors in the master.

A backplane that comes out constant is not stored as an array. Its value goes in
``summary.py`` alongside the arrays, and a comparison against a constant reads it from
there. The derivative suites, enabled whenever the undersampling exceeds one or
``--derivs`` is given, compare the analytic ``d_dlos`` chained through ``dlos_duv``
with the central difference between the offset backplanes; the step is chosen so that
the difference clears the noise floor the iterative photon solver leaves.

Each comparison logs one line: the suite, a status (``Success``, ``Value mismatch``,
``Mask mismatch``, ``Value/mask mismatch``, ``Shape mismatch``, ``No gold master``,
``Invalid gold master``), the title, the value range, the largest difference against the
limit, the offset used against the radius, and the count of failing pixels. Under
pytest the run ends in an ``AssertionError`` if any comparison failed. ``--ignore-missing``
turns a missing master into a warning, for work in progress.

Using it while developing
~~~~~~~~~~~~~~~~~~~~~~~~~

Run one instrument's comparison from the command line while iterating; it is faster
than the pytest wrapper and takes options::

    export PYTHONPATH=.
    python tests/hosts/cassini/iss/gold_master.py                      # compare
    python tests/hosts/cassini/iss/gold_master.py --suite ring pole    # a subset
    python tests/hosts/galileo/ssi/gold_master.py --name C0349632100R  # one observation

A comparison of one Cassini image takes about fifteen seconds at the default
undersampling. To see what a failing backplane looks like, add ``--arrays --browse``
(or ``--debug``, which also writes a log) and open the PNG under the output directory.
``--du`` and ``--dv`` shift the meshgrid by a fraction of a pixel, which shows how
sensitive a comparison is to pointing.

When a change legitimately alters the geometry, or adds a backplane, new masters are
adopted. Never adopt over the real masters in one step: adopt into a scratch directory,
compare against it, inspect the differences from the old set, and only then copy::

    python tests/hosts/cassini/iss/gold_master.py --adopt --gold-master=/tmp/new_masters
    python tests/hosts/cassini/iss/gold_master.py --gold-master=/tmp/new_masters
    pytest tests/hosts/cassini/iss --gold-master=/tmp/new_masters

``--adopt`` runs every suite at full resolution, writes the arrays and browse images
under ``<dir>/<mission>.<instrument>/<basename>/`` with Linux-safe file names whatever the
platform, and writes ``summary.py``. Naming the directory is what keeps it away from the
real tree, which the same command without ``--gold-master`` would overwrite in place.
The tree may be a cloud path such as ``gs://rms-oops-resources/gold_master``.

The unit tests of the framework itself are in ``tests/programs``; they need no
resources.

Coverage
--------

``--cov`` is on by default and measures ``src`` and ``programs``. The check script does
not enforce a threshold, but a change is expected to keep the coverage of the code it
touches, and a new class or backplane arrives with its tests.
