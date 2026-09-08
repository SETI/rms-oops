Surfaces, bodies and gravity
============================

Overview
--------

A :class:`~oops.Surface` is the geometric model of a target; a :class:`~oops.Body` is a
named target with its path, frame, surface and gravity; a :class:`~oops.Gravity` supplies
the orbital frequencies that ring and orbit surfaces need; a
:class:`~oops.lightsource.LightSource` is a source of illumination that shares the body
registry. Their API reference is on the :doc:`oops package </oops>`,
:doc:`oops.surface </oops_surface>`, :doc:`oops.gravity </oops_gravity>` and
:doc:`oops.lightsource </oops_lightsource>` pages.

Surface
-------

A surface is anchored to a path (``origin``, a waypoint) and a frame (``frame``, a
wayframe) and is described in that frame. Four methods define it, and every subclass
must implement all four:

``coords_from_vector3(pos, *, obs=None, time=None, axes=2, derivs=False, hints=None)``
    The surface coordinates of a position: two or three of them, depending on ``axes``.
``vector3_from_coords(coords, *, obs=None, time=None, derivs=False, hints=None)``
    The inverse.
``intercept(obs, los, *, time=None, direction='dep', derivs=False, guess=None, hints=None)``
    Where a line of sight from an observer position meets the surface: the position and
    the parameter *t* such that the intercept is ``obs + t * los``. ``direction`` chooses
    the near or far intercept of a closed surface.
``normal(pos, *, obs=None, time=None, derivs=False, hints=None)``
    The outward normal at a position; its length is arbitrary.

``obs`` and ``time`` matter only for a *virtual* surface, one that exists only from the
viewpoint of an observer, such as a limb or an ansa, or a time-dependent one; the rest
ignore them. ``hints`` is data carried from one call to the next; a caller passes
``hints=True`` to ask for it without having any. Optional overrides are
``intercept_with_normal``, ``intercept_normal_to``, ``velocity`` (the surface velocity
at a point, ``Vector3.ZERO`` by default), ``position_is_inside`` and ``reference``. The
methods that convert whole events, ``coords_of_event``, ``apply_coords_to_event`` and
``event_at_coords``, are built on the four and rarely need overriding.

Each subclass sets the class attributes that describe it. ``COORDINATE_TYPE`` is one of
``"rectangular"``, ``"cylindrical"``, ``"spherical"``, ``"polar"`` or ``"limb"``, and it
is load-bearing: the backplane engine branches on it to decide which family of methods
applies. ``COORDINATE_NAMES``, ``COORDINATE_ABBREVS`` and ``COORDINATE_RANGES`` describe
the coordinates; ``IS_VIRTUAL``, ``IS_TIME_DEPENDENT`` and ``HAS_INTERIOR`` are flags.
The constructor sets four instance attributes: ``origin``, ``frame``, ``unmasked`` (the
same surface without any radial or other limits, or ``self``) and ``intercept_key``, a
hashable tuple that identifies the geometry alone. Two surfaces that differ only in a
mask or a coordinate convention share an intercept key, and the backplane engine uses
it to solve the photon intercept once for all of them.

The photon solver bound onto the class provides ``photon_to_event`` and
``photon_from_event`` (by line of sight), ``photon_to_coords`` and
``photon_from_coords`` (to a point at given surface coordinates),
``photon_normal_to_event`` and ``photon_event_to_normal`` (along the normal) and
``photon_path_to_normal`` and ``photon_normal_to_path``. Each returns a pair of events
with the photon directions, light times and ``perp`` filled in.

The concrete surfaces are :class:`~oops.surface.Spheroid` and
:class:`~oops.surface.Ellipsoid` with their planetocentric and planetographic variants
:class:`~oops.surface.CentricSpheroid`, :class:`~oops.surface.GraphicSpheroid`,
:class:`~oops.surface.CentricEllipsoid` and :class:`~oops.surface.GraphicEllipsoid`;
:class:`~oops.surface.RingPlane`, a plane with optional radial limits, elevation and
radial modes, taking a :class:`~oops.Gravity` for the orbital velocity at each radius;
:class:`~oops.surface.OrbitPlane`, an inclined, eccentric ring; the virtual
:class:`~oops.surface.Ansa`, :class:`~oops.surface.Limb` and
:class:`~oops.surface.PolarLimb`; and :class:`~oops.surface.NullSurface`. The function
:func:`~oops.surface.spice_shape` builds the right spheroid or ellipsoid from the radii
in a SPICE PCK.

Surfaces are not registered anywhere; they are plain objects held by bodies and looked
up through them.

Body
----

A :class:`~oops.Body` binds a name to everything the library knows about a target:
``path`` and ``frame`` (a waypoint and a wayframe), ``surface``, ``radius`` and
``inner_radius`` (the enclosing and enclosed spheres, used by the inventory), ``gravity``,
``parent`` and ``barycenter``, ``children``, ``keywords`` (``"PLANET"``, ``"SATELLITE"``,
``"REGULAR"``, ``"IRREGULAR"``, ``"RING"`` and so on, which ``select_children`` filters
on), and for a planet with rings the ``ring_body``, the registered body whose surface is
the unbounded ring plane. Ring bodies such as ``SATURN_MAIN_RINGS`` are bodies whose
surface is a bounded :class:`~oops.surface.RingPlane` and whose ``is_ring`` is true.

The class holds the registry: ``Body.BODY_REGISTRY`` maps upper-case names to bodies,
``lookup`` and ``as_body`` read it and raise ``KeyError`` for an unknown name, and
``reset_registry`` clears it along with the path and frame registries.
:meth:`~oops.Body.define_solar_system` populates it from SPICE through :mod:`spicedb`:
the leap seconds, the planetary constants, the ephemerides covering the requested time
range, then one body per planet, satellite and barycenter with its
:class:`~oops.path.SpicePath` and :class:`~oops.frame.SpiceFrame`, the ring bodies and
ring frames of the ringed planets, the gravity fields, and the ``SOLAR_DISK`` light
source. The per-planet definitions are private helpers in ``body.py``, and the module
constants there record the ring boundaries. The lower-level ``define_body``,
``define_ring``, ``define_orbit`` and ``define_small_body`` add one body at a time.

Two invariants:

* SPICE gives a later-furnished kernel precedence. Calling ``define_solar_system`` twice
  with different time ranges in one process leaves both sets loaded, and geometry then
  comes from the second; the host tests undefine the solar system between observations
  for this reason (``Body._undefine_solar_system`` unloads every kernel).
* A body's ``photon_to_event`` delegates to its path, so a body and a light source can
  be used interchangeably as a source of illumination.

Gravity
-------

:class:`~oops.Gravity` is the abstract gravity field of a planet. A subclass implements
the potential and the three orbital frequencies at a semimajor axis, ``omega`` (mean
motion), ``kappa`` (radial) and ``nu`` (vertical), their derivatives, ``combo`` and
``solve_a``, the inverse. The derived precession rates ``dmean_dt``, ``dperi_dt`` and
``dnode_dt`` and the resonance helpers ``ilr_pattern`` and ``olr_pattern`` need no
override. The one concrete class, :class:`~oops.gravity.OblateGravity`, takes *GM*, the
zonal harmonics and the reference radius, and its module defines the standard bodies as
both entries in ``Gravity.GRAVITY_REGISTRY`` and class attributes (``Gravity.SATURN``,
``Gravity.PLUTO_CHARON``); ``Gravity.lookup`` reads the registry.

LightSource
-----------

A :class:`~oops.lightsource.LightSource` is a named source of photons: a path, for a
source that moves, or a fixed J2000 direction given as a vector or as right ascension
and declination in degrees. Its ``photon_to_event`` fills in the arrival direction of an
event the way a path's does, and :class:`~oops.lightsource.DiskSource` adds a disk of
finite angular size. Light sources are stored in ``Body.BODY_REGISTRY`` under their
names, in the same namespace as bodies, so that a backplane's source key can name either;
the constructor refuses a name a body already has. The subclass is imported at the bottom
of the package module to break the cycle.
