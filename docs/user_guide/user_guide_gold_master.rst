The gold master tool
====================

Purpose
-------

``gold_master`` is a command-line program that computes every standard backplane of an
observation and, in its default mode, compares each against a stored reference array,
the *gold master*. Two uses follow from that. For a user, it is the quickest way to see
the geometric contents of an image: one command writes a couple of hundred backplanes
as arrays and as browse images, with no code. For a developer, it is the regression
test that detects any change in the geometry (see the :doc:`Developer's Guide
</dev_guide/dev_guide_testing>`).

The tool is the ``programs.gold_master`` package of the repository. It is not part of
the ``oops`` library and does not ship in the wheel, so it runs from a checkout with the
repository root on ``PYTHONPATH``. Each instrument that has reference arrays has a
runnable module under ``tests/hosts``::

    export PYTHONPATH=.
    python tests/hosts/cassini/iss/gold_master.py --help
    python tests/hosts/galileo/ssi/gold_master.py --help
    python tests/hosts/juno/junocam/gold_master.py --help

Each module registers the instrument's *standard observations*, files in the test data
tree with the bodies to test, and names the host module whose ``from_file`` reads them.
With no arguments the program compares every standard observation of the instrument
against its masters.

Previewing the geometry of an image
-----------------------------------

To see what is in an image, run a *preview*: the program computes every backplane at
full resolution, writes each as a pickle file and as a PNG, and compares nothing. For
one of the standard observations::

    python tests/hosts/cassini/iss/gold_master.py --preview --name W1573721822_1 -o /tmp/preview

For an image of your own, give its path and name the host module, the planet, and any
moons and rings to include::

    python tests/hosts/cassini/iss/gold_master.py /data/N1573845439_1.IMG \
        --module oops.hosts.cassini.iss --planet SATURN --ring SATURN_MAIN_RINGS \
        --preview -o /tmp/preview

The planet, moon and ring names are registered body names; the instrument module's
default targets do not apply to a file given by path, so at least one of ``--planet``,
``--moon`` or ``--ring`` is required. Every body gets its surface and limb; a planet
with rings gets its ring plane and ansa as well; and a named ring gets its own bounded
ring plane and ansa. ``--suite`` restricts the run to some families of backplanes, which
is worthwhile at full resolution: a full preview of a Cassini image takes about two
minutes, a preview of the ring and sky backplanes alone about twenty seconds.

The output lands under the directory given by ``-o`` (or
``$OOPS_BACKPLANE_OUTPUT_PATH``, or the current directory), in a directory named for the
mission and instrument and one named for the file:

::

    /tmp/preview/cassini.iss/N1573845439_1/
    ├── arrays/
    │   ├── SATURN-RING radius (km).pickle
    │   ├── SATURN_MAIN_RINGS longitude wrt node (deg).pickle
    │   └── ...                     one pickle per backplane with a shaped value
    ├── browse/
    │   ├── SATURN-RING radius (km).png
    │   └── ...                     one image per array
    ├── summary.py                  the range of every backplane, and the value of the
    │                               ones that are constant
    └── preview.log                 written with --log

Each pickle holds the ``polymath`` array itself, values and mask together, with the
angles converted to degrees as the title says:

.. code-block:: python

    import pickle
    with open('/tmp/preview/cassini.iss/N1573845439_1/arrays/SATURN-RING radius (km).pickle', 'rb') as f:
        radius = pickle.load(f)
    radius.vals[radius.antimask]

Each browse image is an 8-bit greyscale PNG scaled to the range of the unmasked values,
with masked pixels black; for a Boolean backplane, False is dark grey and True white.
The image is oriented with *v* increasing downward, as the data array is displayed.

.. list-table::
   :widths: 33 33 33
   :header-rows: 1

   * - Ring radius
     - Incidence angle on Saturn
     - Where Epimetheus is intercepted
   * - .. image:: images/saturn_ring_radius.png
     - .. image:: images/saturn_incidence_angle.png
     - .. image:: images/epimetheus_where_intercepted.png

These three are from the standard Cassini image W1573721822_1, reduced in size. The
radius runs from black at the inner edge of the field to white at the outer, the
incidence angle is masked black off the planet and brightest toward the terminator, and
the one-pixel moon is the white dot.

``summary.py`` is a Python literal, a dictionary from each backplane's title to its
range. Its header explains the tuple in each entry: a Boolean backplane gives the counts
of False, True, masked and total pixels; a floating-point one gives the minimum, maximum,
masked count and total; a constant one gives the value in place of the range; and a
fully masked one gives ``None``. It can be read with ``eval`` or ``ast.literal_eval``
after skipping the comment lines, and it is the quickest summary of what an image
contains.

The file names are the backplane titles made safe for the platform: on macOS and
Windows the spaces and parentheses stay and the colon becomes a hyphen, and on Linux,
and in every set of reference masters, spaces become underscores and all other
punctuation is dropped or replaced, so that ``SATURN:RING radius (km)`` is written as
``SATURN-RING radius (km).png`` on a Mac and ``SATURN-RING_radius_km.png`` on Linux.

Comparing against the masters
-----------------------------

The default task, ``--compare``, computes the backplanes on a grid undersampled by 16
and compares each with its master, logging one line per comparison::

    2026-09-07 18:28:02.279 | programs.gold_master | INFO  | ansa | Success: "SATURN:ANSA radius (km)"; min,max=4912.,1.28e+05; diff=2.365e-07/0.1, pixels=0/4096

The fields are the suite, the status, the title, the range of the values (and the
masked fraction, when any are masked), the largest difference over the limit, and the
number of failing pixels over the total. A failing comparison has the status
``Value mismatch``, ``Mask mismatch``, ``Value/mask mismatch`` or ``Shape mismatch``,
and ``No gold master`` means the master is missing, which ``--ignore-missing`` reduces
to a warning. The run ends with the counts of warnings and errors and the elapsed
time, about fifteen seconds for one Cassini image, and exits non-zero if anything
failed. Under pytest the same comparison runs as ``pytest tests/hosts``.

``--gold-master DIR`` compares against masters in another directory, which must have the
same layout, and ``--adopt`` writes a new set of masters into it. Adopt only into a
directory named this way: without ``--gold-master``, the command overwrites the real
masters in place.

Options
-------

The invocation is::

    python tests/hosts/<mission>/<instrument>/gold_master.py [filepath ...] [options]

Data objects
~~~~~~~~~~~~

``filepath``
    One or more data files to use in place of a standard observation. Positional;
    cannot be combined with ``--name``.
``--index N``
    The index to use when the host's ``from_file`` returns a list; otherwise backplanes
    are generated for each observation in the file.
``--module NAME``
    The module whose ``from_file`` reads the files, such as ``oops.hosts.cassini.iss``.
    Default is the module the instrument's ``gold_master.py`` registered.
``--name NAME [NAME ...]``, ``-n``
    The standard observations to use when no file is given. Default is all of them.

Backplane targets
~~~~~~~~~~~~~~~~~

``--planet NAME [NAME ...]``, ``-p``
    The planets to generate backplanes for. Default comes from the standard observation.
``--moon NAME [NAME ...]``, ``-m``
    The moons. Default likewise.
``--ring NAME [NAME ...]``, ``-r``
    The rings. Backplanes are always generated for the full ring plane of each planet;
    this adds bounded rings such as ``SATURN_MAIN_RINGS``.

Testing options
~~~~~~~~~~~~~~~

``--preview``
    Generate the backplane arrays and browse images without comparing. Forces
    ``--arrays``, ``--browse`` and full resolution.
``--compare``, ``-c``
    Generate and compare. The default.
``--adopt``, ``-a``
    Write the arrays as the new gold masters, overwriting the existing ones. Forces
    full resolution and every suite.
``--debug``
    Shorthand for ``--arrays --browse --log``.
``--tolerance TOL``
    A factor applied to every comparison's tolerance. Default 1.
``--radius RAD``
    A factor applied to every comparison's radial offset limit, the number of pixels by
    which a value or mask edge may be shifted and still pass. Default 1.
``--ignore-missing``
    Log a warning rather than an error when a gold master is missing.
``--suite NAME [NAME ...]``
    The suites to run: ``ansa``, ``border``, ``distance``, ``lighting``, ``limb``,
    ``orbit``, ``pole``, ``resolution``, ``ring``, ``sky``, ``spheroid``, ``where``.
    Default is all of them; ``--adopt`` always runs all of them.
``--du DU``, ``--dv DV``
    An offset, in pixels, applied to the meshgrid origin, for testing the sensitivity to
    pointing. Default 0.
``--derivs``, ``--no-derivs``
    Whether to test the spatial derivatives of the backplanes against finite
    differences. Default is to test them when undersampling, and not at full
    resolution; the tests take several times longer with them.

Backplane array options
~~~~~~~~~~~~~~~~~~~~~~~

``--arrays``, ``--no-arrays``
    Whether to save the arrays as pickles. Default is not to, except under ``--preview``
    and ``--adopt``.
``--undersample N``, ``-u``
    The factor by which to undersample the grid. Default 16; forced to 1 by
    ``--preview`` and ``--adopt``.
``--inventory``, ``--no-inventory``
    Whether to use a body inventory to confine each body's calculation to its bounding
    box. Default is to, except where the instrument's module says otherwise (Galileo SSI
    and JunoCam turn it off, because their wide fields make the box unreliable).
``--border N``
    Pixels by which to widen each inventory box. Default 0.
``--save-sampled``, ``--ss``
    Also save the masters resampled at the undersampled grid points, for a direct
    comparison of the two arrays.

Browse image options
~~~~~~~~~~~~~~~~~~~~

``--browse``, ``--no-browse``
    Whether to save browse images. Default is not to, except under ``--preview`` and
    ``--adopt``.
``--zoom N``
    A zoom factor for the browse images, by pixel replication. Default 1.
``--format EXT``
    ``png`` (the default), ``jpg`` or ``tiff``.

Output options
~~~~~~~~~~~~~~

``--gold-master DIR``, ``-g``
    The root of the gold master files. Default is ``$OOPS_GOLD_MASTER_PATH``, or the
    ``gold_master`` subdirectory of ``$OOPS_RESOURCES``. The directory must have the
    standard layout, ``<dir>/<mission>.<instrument>/<basename>``.
``--output DIR``, ``-o``
    The root for saved arrays, browse images and logs. Default is
    ``$OOPS_BACKPLANE_OUTPUT_PATH``, or the current directory.
``--verbose``, ``-v``; ``--quiet``, ``-q``
    Whether to write the log to the terminal. Default is to.
``--log``, ``--no-log``
    Whether to write a log file, named for the task, in the output directory. Default
    is not to.
``--level LEVEL``
    The minimum level to log: ``debug``, ``info``, ``warning``, ``error`` or an integer
    from 1 to 30. Default ``debug``.
``--convergence``
    Show the iterations of the photon solvers in the log.
``--diagnostics``
    Include diagnostic information: the interpolation windows created, the cached
    intercepts reused, and the time collapses.
``--internals``
    Dump the backplane's internal caches at the end of the log.
``--performance``
    Include the wall time of each photon solution.
``--fullpaths``
    Include the full path of every output file in the log.
``--platform OS``
    ``macos``, ``windows`` or ``linux``, to name the output files as that platform would.
    Default is the platform the program runs on; gold master files always use Linux
    names.

The log
-------

A log line has the form ``<time> | programs.gold_master | <level> | <suite> | <message>``,
with the level one of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` and ``FATAL``. Under
``--preview`` each backplane is logged as ``Written`` with its range; under ``--compare``
each comparison is logged as described above. With ``--log`` the same lines go to
``<output>/<mission>.<instrument>/<basename>/<task>.log``, and an existing log is
renamed with its modification time rather than overwritten.

From Python
-----------

The same run can be driven from a short script, which is how an instrument's
``gold_master.py`` works and how an observation outside the test data tree is
registered with defaults of its own:

.. code-block:: python

    import programs.gold_master as gm

    gm.define_standard_obs('my_image',
            obspath = '/absolute/path/to/my_image.IMG',
            index   = None,
            planets = ['SATURN'],
            rings   = ['SATURN_MAIN_RINGS'])
    gm.set_default_args(module='oops.hosts.cassini.iss', inventory=False)

    if __name__ == '__main__':
        gm.execute_as_command()

An absolute ``obspath`` is used as it is; a relative one is joined to
``$OOPS_TEST_DATA_PATH``. Running the script with ``--preview --name my_image`` then
writes the backplanes as above. The :doc:`API reference </gold_master>` documents
:func:`~programs.gold_master.define_standard_obs`,
:func:`~programs.gold_master.set_default_args` and the rest.
