Extending the library
=====================

Each recipe below names the base class, the methods to implement, where the file goes
and what else has to change. The last section is the checklist that applies to every
addition, whichever family it belongs to.

Adding a Path
-------------

Put the class in ``src/oops/path/<name>.py``, lowercase with no separators. Inherit
:class:`~oops.Path`, set the three required attributes, register, and implement
``event_at_time``. :class:`~oops.path.LinearPath` is the model:

.. code-block:: python

    ##########################################################################################
    # oops/path/linearpath.py
    ##########################################################################################

    from polymath   import Qube, Scalar, Vector3
    from oops.event import Event
    from oops.frame import Frame
    from oops.path  import Path


    class LinearPath(Path):
        """A Path subclass describing linear motion relative to another Path and Frame."""

        _WAYPOINTS = {}

        def __init__(self, pos, epoch, origin, *, frame=None, path_id=None):
            """Constructor for a LinearPath.

            Parameters:
                pos (Vector3Like): Position at the epoch, with the velocity as its "t"
                    derivative, or a tuple (position, velocity).
                epoch (ScalarLike): Time in seconds TDB at which the position applies.
                origin (Path | str): The path or ID relative to which this one is defined.
                frame (Frame | str, optional): The frame; default is the origin's frame.
                path_id (str, optional): The ID under which to register the path.
            """

            ...                       # store the parameters as private attributes

            # Required attributes
            self._origin = Path.as_waypoint(origin)
            self._frame = frame and Frame.as_wayframe(frame) or self._origin._frame
            self._shape = Qube.broadcasted_shape(self._pos, self._vel, self._epoch,
                                                 self._origin._shape, self._frame._shape)

            self._register(path_id)
            self.refresh()

        def _waypoint_key(self):
            ...                       # a hashable tuple of the defining parameters

        def __getstate__(self):
            ...                       # the constructor arguments, ending with stripped_id

        def __setstate__(self, state):
            self.__init__(*state)

        def event_at_time(self, time, *, quick=None):
            time = Scalar.as_scalar(time)
            pos = self._pos + (time - self._epoch) * self._vel
            return Event(time, (pos, self._vel), self._origin, self._frame)

    Path._PATH_SUBCLASSES.append(LinearPath)

    ##########################################################################################

Notes on the pattern:

* ``_WAYPOINTS`` and ``_waypoint_key`` make instances with equal parameters share one
  registered waypoint. Omit both if every instance should be its own waypoint.
* ``__getstate__`` returns the constructor arguments as a tuple so that pickling and
  :meth:`~oops.Fittable.copy` can rebuild the object; a registered class ends the tuple
  with ``self.stripped_id``, which ``copy`` drops so the copy stays unregistered.
* Set ``_USE_QUICKPATHS = True`` if evaluating the path is slow and it varies smoothly,
  so that ``quick_path`` may interpolate it.
* A Fittable path additionally inherits :class:`~oops.Fittable`, defines ``nparams``,
  the ``params`` property and ``_set_params``, and drops its cached state in ``_refresh``.
  :class:`~oops.path.PathShift` is the model.
* The last statement appends the class to ``Path._PATH_SUBCLASSES`` so that
  ``Path._reset_caches`` can clear its waypoints.

Then export it: add the import and the ``__all__`` entry to ``src/oops/path/__init__.py``
and declare the class in ``src/oops/path/__init__.pyi``. The API page
:doc:`oops.path </oops_path>` picks it up from ``__all__`` with no further change.

Adding a Frame
--------------

Identical in structure, in ``src/oops/frame/<name>.py``. Inherit :class:`~oops.Frame`,
set ``_reference`` (a wayframe), ``_origin`` (a waypoint, or ``None`` for an inertial
frame) and ``_shape``, call ``_register(frame_id)`` and ``refresh()``, and implement
``transform_at_time``, which returns a :class:`~oops.Transform`. A class sharing
wayframes declares ``_WAYFRAMES = {}`` and ``_wayframe_key``; a time-varying frame sets
``_USE_QUICKFRAMES = True``; the module ends with
``Frame._FRAME_SUBCLASSES.append(<Class>)``. :class:`~oops.frame.SpinFrame` is the
smallest complete model:

.. code-block:: python

    def transform_at_time(self, time, *, quick=None):
        time = Scalar.as_scalar(time)
        angle = self._rate * (time - self._epoch) + self._offset
        matrix = Matrix3.z_rotation(angle)        # or the axis the frame spins about
        omega = self._rate * Vector3.ZAXIS
        return Transform(matrix, omega, self, self._reference, self._origin)

Export it from ``src/oops/frame/__init__.py`` and declare it in ``__init__.pyi``.

Adding a Surface
----------------

In ``src/oops/surface/<name>.py``, inherit :class:`~oops.Surface`, set the class
attributes that describe the coordinate system, and implement the four abstract methods.
:class:`~oops.surface.RingPlane` shows every piece; the skeleton is:

.. code-block:: python

    class MySurface(Surface):
        """One-line description."""

        COORDINATE_TYPE = 'polar'                  # or spherical, cylindrical, ...
        COORDINATE_NAMES = ('radius', 'longitude', 'elevation')
        COORDINATE_ABBREVS = ('r', 'theta', 'z')
        COORDINATE_RANGES = ((0, None), (0, TWOPI), (None, None))
        IS_VIRTUAL = False
        IS_TIME_DEPENDENT = False
        HAS_INTERIOR = False

        def __init__(self, origin, frame, ...):
            self.origin = Path.as_waypoint(origin)
            self.frame = Frame.as_wayframe(frame)
            ...
            self.unmasked = self                   # or the same surface without limits
            self.intercept_key = ('mysurface', self.origin.waypoint, self.frame.wayframe,
                                  <the parameters that change the geometry>)

        def __getstate__(self): ...
        def __setstate__(self, state): ...

        def coords_from_vector3(self, pos, *, obs=None, time=None, axes=2, derivs=False,
                                hints=None): ...
        def vector3_from_coords(self, coords, *, obs=None, time=None, derivs=False,
                                hints=None): ...
        def intercept(self, obs, los, *, time=None, direction='dep', derivs=False,
                      guess=None, hints=None): ...
        def normal(self, pos, *, obs=None, time=None, derivs=False, hints=None): ...

Every method must honor ``derivs``: when it is true, the returned quantities carry the
derivatives of their inputs, which is what lets a backplane report its gradient. Return
``hints`` when the caller passed any, appended to the tuple. Override ``velocity`` if
the surface moves within its frame (a ring plane does, by orbital motion) and
``position_is_inside`` if ``HAS_INTERIOR`` is true. The photon solvers come for free
from the base class.

Export from ``src/oops/surface/__init__.py`` and the stub. If a backplane should be able
to reach the surface by a key, the key must resolve to it through a :class:`~oops.Body`:
either a new body whose ``surface`` it is, or a new suffix handled in
``Backplane.get_surface`` and ``_get_body_and_modifier``.

Adding an FOV, a Cadence or a Calibration
-----------------------------------------

The same shape, with fewer moving parts because none of the three is registered:

* An :class:`~oops.FOV` subclass sets ``uv_shape``, ``uv_los``, ``uv_scale`` and
  ``uv_area`` in its constructor, each :class:`~oops.Pair` made read-only, and implements
  ``xy_from_uvt`` and ``uv_from_xyt`` honoring ``derivs`` and ``remask``. A subclass with
  its own ``_refresh`` calls ``super()._refresh()``. :class:`~oops.fov.FlatFOV` is the
  model, and :class:`~oops.fov.Platescale` the model for a Fittable one.
* A :class:`~oops.Cadence` subclass sets the eight ``# Required attributes`` and
  implements the six abstract methods; :class:`~oops.cadence.Metronome` is the model.
* A :class:`~oops.Calibration` subclass sets ``name``, ``factor``, ``baseline``,
  ``has_baseline``, ``shape`` and ``fov`` and implements the five abstract methods;
  :class:`~oops.calibration.FlatCalib` is the model, and ``prescale`` is usually one line
  through ``_prescaled_args``.

Each is exported from its subpackage's ``__init__.py`` and declared in the stub.

Adding a backplane
------------------

A new backplane is a function in an existing module of ``src/oops/backplane`` when it
belongs to a family that is already there, or a new module when it does not. Either way
it follows the pattern in :doc:`dev_guide_backplanes`:

.. code-block:: python

    def my_quantity(self, event_key, option='default'):
        """One-line noun phrase describing the quantity and its units.

        Parameters:
            event_key (str | tuple): Key defining the surface event.
            option (str, optional): What the option selects. Default is 'default'.

        Returns:
            Scalar: The quantity in km, registered as a backplane.

        Raises:
            ValueError: If `option` is not one of the recognized values.
        """

        if option not in ('default', 'other'):
            raise ValueError('invalid option: ' + repr(option))

        self.refresh()
        event_key = Backplane.standardize_event_key(event_key)
        key = ('my_quantity', event_key, option)
        if key in self._backplanes:
            return self.get_backplane(key)

        event = self.get_surface_event(event_key)
        value = ...                                # from the event's properties
        return self.register_backplane(key, value)

Rules:

* The cache key is the method name, the standardized event key, and the remaining
  arguments in signature order, so that :meth:`~oops.Backplane.evaluate` can rebuild
  the call from the key.
* A method whose result does not depend on the pixel calls itself with
  ``Backplane.gridless_event_key(event_key)``; name such methods ``center_*`` or
  ``sub_*`` as the existing ones do.
* Keep the module free of anything that is not meant to become a method. A shared helper
  is a private function that takes ``self``.
* A new module needs its import in ``src/oops/backplane/all.py`` (alphabetical) and the
  banner naming it.

Then, in the same change: add the method's signature to
``src/oops/backplane/__init__.pyi`` in alphabetical position among the backplane methods,
returning ``Any``; add a test in ``tests/backplane/`` against the synthetic Saturn
observation that the package's ``conftest.py`` provides; and, if the quantity should be
regression-tested against real observations, add it to the matching gold master suite in
``programs/gold_master`` and re-adopt the masters (see :doc:`dev_guide_testing`). Add a
row to the table in the :doc:`User's Guide </user_guide/user_guide_backplanes>`.

Adding a gold master test suite
-------------------------------

A suite is a module in ``programs/gold_master`` with one function that receives a
``BackplaneTest`` and calls its ``gmtest`` and ``compare`` methods, followed by a
registration call. ``pole.py`` is the whole pattern:

.. code-block:: python

    from programs.gold_master import register_test_suite

    def pole_test_suite(bpt):
        """Test the pole clock angle and pole position angle of every body and ring.
        ...
        """

        bp = bpt.backplane
        for name in bpt.body_names + bpt.ring_names:

            clock = bp.pole_clock_angle(name)
            position = bp.pole_position_angle(name)
            bpt.gmtest(clock,
                       name + ' pole clock angle (deg)',
                       method='mod360', limit=0.001, radius=1)
            bpt.gmtest(position,
                       name + ' pole position angle (deg)',
                       method='mod360', limit=0.001, radius=1)
            bpt.compare(clock + position, 0.,
                        name + ' pole clock plus position angle (deg)',
                        method='mod360', limit=1.e-13, radius=1)

    register_test_suite('pole', pole_test_suite)

Add the import to ``programs/gold_master/all.py`` and the suite name to the list in
``docs/gold_master.rst``. :doc:`dev_guide_testing` explains the test object and the
comparison options.

The checklist for every addition
--------------------------------

#. **Source.** The file has the banner, is under 90 columns, and follows the docstring
   style of its neighbors, with every parameter and return typed in the docstring.
#. **Export.** The subpackage ``__init__.py`` imports the name and lists it in
   ``__all__``.
#. **Stub.** ``__init__.pyi`` declares the class outright, with every public method's
   signature and the docstring types translated to annotations. Any name that exists
   only at run time goes in ``stubtest-allowlist.txt``. ``run-all-checks.sh --stubtest``
   confirms the two agree.
#. **Tests.** A ``tests/<subpackage>/test_<name>.py`` mirroring the source, using the
   ``core_kernels`` fixture when SPICE geometry is involved.
#. **Documentation.** A class in an existing subpackage appears on that subpackage's API
   page through ``__all__``. A new top-level class needs an ``autoclass`` line in
   ``docs/oops.rst``, kept alphabetical, and a new subpackage needs its own
   ``docs/oops_<name>.rst`` and a line in the ``index.rst`` toctree. The relevant chapter
   of this guide and of the User's Guide is updated in the same change. Both Sphinx
   builds pass with ``-W -n``.
#. **Checks.** ``./scripts/run-all-checks.sh`` passes.
