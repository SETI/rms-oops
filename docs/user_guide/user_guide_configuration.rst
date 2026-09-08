Configuration
=============

Where settings come from
------------------------

There is no configuration file. Three things configure a session, in this order of
precedence:

#. **Arguments** to the call at hand: the ``quick`` and ``converge`` keywords that the
   geometry methods accept, the options of :meth:`~oops.Body.define_solar_system`, and
   the keywords of a host's ``from_file``.
#. **The settings in** :mod:`oops.config`, module-level classes whose attributes are the
   defaults for those arguments and the switches for logging. Assign to an attribute to
   change it for the rest of the session.
#. **Environment variables**, which locate the resources (:doc:`user_guide_installation`).

Interpolation: QUICK
--------------------

Evaluating a SPICE ephemeris or pointing at a million slightly different times is slow,
so the library interpolates each path and frame over the time window it needs. The
:class:`~oops.config.QUICK` class holds the parameters in its ``dictionary``: the sampling
step (``path_time_step`` and ``frame_time_step``, 0.05 s), the margin added at each end
of a window (``path_time_extension``, 5 s), the number of windows cached per object
(``quickpath_cache_size``, 40) and switches to disable each kind
(``use_quickpaths``, ``use_quickframes``). ``QUICK.flag`` turns the whole mechanism off
when set to ``False``.

Every method that evaluates a path or frame takes a ``quick`` keyword. ``None``, the
default, uses the dictionary; a dictionary of your own overrides individual entries for
that call; ``False`` bypasses interpolation and evaluates SPICE directly, which is exact
and slow. Do not pass ``True``: it is not a dictionary, so it disables the interpolation
rather than requesting it.

Convergence: PATH_PHOTONS and SURFACE_PHOTONS
---------------------------------------------

Finding where a photon left a moving body, or where a line of sight meets a moving
surface, is an iteration on the light travel time. :class:`~oops.config.PATH_PHOTONS`
and :class:`~oops.config.SURFACE_PHOTONS` hold the limits: ``max_iterations`` (4 for
paths, 6 for surfaces), ``dlt_precision`` (the change in light time, in seconds, below
which iteration stops), ``dlt_limit`` (the largest permitted departure from the initial
estimate), and the ``km_precision`` and ``rel_precision`` goals. The ``converge`` keyword
of the photon methods takes a dictionary overriding any of them for one call. The
defaults reach ten-centimeter precision and rarely need changing; raising
``max_iterations`` is the first thing to try if a warning reports non-convergence.

:class:`~oops.config.EVENT_CONFIG` ``.collapse_threshold`` (3 s) is the spread of times
below which an event's time array is replaced by a single value, which keeps the
interpolation windows short for an observation whose pixels differ in time by
milliseconds.

Logging
-------

:class:`~oops.config.LOGGING` controls what the library reports and where. By default it
writes warnings and errors to standard output and nothing else. Its static methods
change that:

.. code-block:: python

    from oops.config import LOGGING

    LOGGING.all(True, category='convergence')    # log every solver iteration
    LOGGING.all(True, category='diagnostics')    # log interpolation windows and time collapse
    LOGGING.all(True)                            # both
    LOGGING.off()                                # back to warnings and errors only

    LOGGING.set_file('oops.log')                 # copy the log to a file
    LOGGING.set_logger(my_logger, level='INFO')  # route it through a logging.Logger
    LOGGING.set_stdout(False)                    # silence the terminal (needs another sink)

The individual switches, ``path_iterations``, ``surface_iterations``,
``fov_iterations``, ``observation_iterations``, ``quickpath_creation``,
``quickframe_creation``, ``event_time_collapse`` and ``surface_time_collapse``, are
attributes that can be set one at a time. ``push`` and ``pop`` save and restore the
whole configuration around a block of code. The counters ``LOGGING.warnings`` and
``LOGGING.errors`` record how many of each were issued.

Kernel selection
----------------

:meth:`~oops.Body.define_solar_system` decides which SPICE kernels are loaded:

.. code-block:: python

    Body.define_solar_system(start_time, stop_time, asof=None,
                             planets=None, mst_pck=True, irregulars=True)

``start_time`` and ``stop_time`` (dates as strings, or seconds TDB) select the
ephemerides that cover the interval; with neither, the extended long-range ephemerides
are used. ``planets`` restricts the definition to one planet (by number, 1 through 9)
or a tuple of them, which is faster and loads fewer kernels; ``None`` or ``0`` means
all nine. ``asof`` is a date, and only kernels that existed by then are used, so that a
calculation can be repeated with the kernels of an earlier time. ``mst_pck`` includes
the constants kernels that update the rotation of Saturn's small moons; ``irregulars``
includes the irregular satellites.

The host modules call this for you with the mission's own time range, and each accepts
``planets``, ``asof`` and the other keywords through its ``initialize`` function or
``from_file`` (see :doc:`user_guide_appendix_hosts`). Two things to know:

* SPICE gives a later-loaded kernel precedence. Calling ``define_solar_system`` again
  with a different range in the same process leaves both loaded, and the geometry then
  comes from the second. To start over, call ``Body.reset_registry()`` after unloading
  through ``spicedb.unload_all()``.
* The Hubble, Webb and Keck host modules define the solar system when they are
  imported, over their mission's whole date range, so importing one has the side
  effect of loading kernels.

Pickling and other switches
---------------------------

:class:`~oops.config.PICKLE_CONFIG` selects how much internal state a pickled path,
frame or backplane carries (``quickpath_details``, ``quickframe_details`` and
``backplane_events``, all ``True``). :class:`~oops.config.AREA_FACTOR` ``.old`` selects an
earlier definition of a pixel's solid angle for comparison with old results.
