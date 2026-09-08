Geometry below the backplanes
=============================

Backplanes answer most questions, but the objects they are built from are usable on
their own: for the geometry of a single point rather than an image, for an instrument
with no host module, or to understand what a backplane computed. This chapter walks
through them with the Saturn system in November 2007. The numbers shown are what the
calls print with the resource tree of the Ring-Moon Systems Node.

Defining the solar system
-------------------------

.. code-block:: python

    import oops
    from oops.body import Body

    kernels = Body.define_solar_system('2007-11-01', '2007-11-30', planets=6)

This furnishes the leap-second kernel, the planetary constants, the ephemerides of the
Saturn system covering the month, and the Cassini-era constants for Saturn's small
moons, and registers a :class:`~oops.Body` for the Sun, the barycenters, Saturn and its
moons and rings. A body holds everything the library knows about a target:

.. code-block:: python

    >>> saturn = Body.lookup('SATURN')
    >>> saturn.spice_id, saturn.radius
    (699, 60268.0)
    >>> saturn.path.path_id, saturn.frame.frame_id
    ('SATURN', 'IAU_SATURN')
    >>> type(saturn.surface).__name__, saturn.ring_body.name
    ('Spheroid', 'SATURN_RING_PLANE')
    >>> saturn.child_names[:6], len(saturn.child_names)
    (['MIMAS', 'ENCELADUS', 'TETHYS', 'DIONE', 'RHEA', 'TITAN'], 68)
    >>> Body.lookup('EPIMETHEUS').keywords
    ['EPIMETHEUS', 'SATELLITE', 'REGULAR', 'SATURN']
    >>> Body.exists('SATURN_MAIN_RINGS')
    True

``select_children`` filters a body's moons by keyword, radius or ring, and
``Body.BODY_REGISTRY`` is the whole registry. Names are case-insensitive on lookup and
upper case in the registry.

Spacecraft need their own kernels. ``define_solar_system`` loads planets and moons
only; a mission's host module furnishes the spacecraft trajectory and pointing when it
initializes, and :mod:`spicedb` does it directly:

.. code-block:: python

    import spicedb

    spicedb.open_db()
    spicedb.furnish_cassini_kernels('2007-11-01', '2007-11-30')
    spicedb.close_db()

Paths and events
----------------

A :class:`~oops.path.SpicePath` is any body in an SPK kernel, relative to an origin and
in a frame; ``event_at_time`` gives its state as an :class:`~oops.Event`:

.. code-block:: python

    >>> import julian
    >>> from oops.path import SpicePath
    >>> tdb = julian.tdb_from_iso('2007-11-15T12:00:00')
    >>> cassini = SpicePath('CASSINI', 'SATURN')      # relative to Saturn, in J2000
    >>> event = cassini.event_at_time(tdb)
    >>> event.pos
    Vector3(841965.7188776  914956.47700812 -34420.5039178 )
    >>> event.vel
    Vector3(-5.88520327 -1.62489315  0.25076181)
    >>> event.origin_id, event.frame_id
    ('SATURN', 'J2000')
    >>> event.pos.norm()
    Scalar(1243879.6)

An event can be re-expressed relative to any registered path and frame:

.. code-block:: python

    >>> event.wrt_ssb().origin_id                      # relative to the barycenter
    'SSB'
    >>> event.wrt(saturn.path, saturn.frame).pos      # in Saturn's rotating frame
    Vector3(-1124806.95907089  -520643.61460127   104766.06833687)

Registered paths and frames are looked up by ID: ``Path.as_path('SATURN')``,
``Frame.as_frame('IAU_SATURN')``, and the roots ``Path.SSB`` and ``Frame.J2000``.
:class:`~oops.frame.SpiceFrame` wraps a frame from a C-kernel or PCK, and
``transform_at_time`` gives the :class:`~oops.Transform` that rotates vectors into it
at a time.

Light-time solutions
--------------------

The photon methods connect two events along a light path. To find where Saturn's
center was when the light reaching Cassini left it:

.. code-block:: python

    >>> from oops.event import Event
    >>> from polymath import Vector3
    >>> arrival = Event(tdb, Vector3.ZERO, cassini, 'J2000')
    >>> (departure, arrival) = saturn.path.photon_to_event(arrival)
    >>> arrival.arr_lt, departure.time
    (Scalar(-4.149), Scalar(248400061.03373036))
    >>> ra, dec = arrival.ra_and_dec(apparent=True)
    >>> ra * oops.DPR, dec * oops.DPR
    (227.3783, 1.5857)

``photon_to_event`` returns the event on the path, with the departing direction and the
light time filled in, and a copy of the arrival event with the arriving direction and
its (negative) light time. ``photon_from_event`` goes the other way. ``ra_and_dec``,
``incidence_angle``, ``emission_angle`` and ``phase_angle`` read those directions;
``apparent=True`` includes stellar aberration. These are the calls a backplane makes for
every pixel at once, with an array-valued ``arr`` direction in place of a single one.

Surfaces
--------

A :class:`~oops.Surface` converts between positions and surface coordinates and
intercepts lines of sight, in its own frame. Saturn's surface is a
:class:`~oops.surface.Spheroid` in the ``IAU_SATURN`` frame:

.. code-block:: python

    >>> surface = saturn.surface
    >>> surface.COORDINATE_NAMES
    ('longitude', 'latitude', 'elevation')
    >>> obs_pos = event.wrt(saturn.path, saturn.frame).pos
    >>> (pos, t) = surface.intercept(obs_pos, -obs_pos.unit())   # look at the center
    >>> pos
    Vector3(54454.5248227  25205.57008156 -5071.96939245)
    >>> lon, lat, elev = surface.coords_from_vector3(pos, axes=3)
    >>> lon * oops.DPR, lat * oops.DPR, elev
    (24.838, -5.353, 0.0)
    >>> surface.normal(pos).unit()
    Vector3( 0.90264046  0.41780857 -0.10332566)

``vector3_from_coords`` is the inverse, and ``photon_to_event`` on a surface solves the
light path from an arrival event to the moving surface, which is the operation behind
every ``where_intercepted``. The ring plane is a :class:`~oops.surface.RingPlane` with
coordinates ``('radius', 'longitude', 'elevation')``; ``saturn.ring_body.surface`` is the
unbounded one and ``Body.lookup('SATURN_MAIN_RINGS').surface`` is bounded to the main
rings. Longitudes on a body are measured from the IAU prime meridian and increase
eastward in these coordinates; the backplane methods offer the westward and other
conventions on top.

Building an observation by hand
-------------------------------

An instrument without a host module can be described directly. This points a flat,
distortion-free camera at Saturn from Earth:

.. code-block:: python

    from oops.fov import FlatFOV
    from oops.frame import Frame, TwoVectorFrame
    from oops.observation import Snapshot
    from oops.path import Path
    from polymath import Scalar, Vector3

    time = 1.e8                                   # seconds TDB
    earth = Path.as_path('EARTH')
    saturn = Body.lookup('SATURN')
    los = saturn.path.event_at_time(Scalar(time)).wrt_path(earth).pos.unit()

    TwoVectorFrame(Frame.J2000, los, 'z', Vector3.XAXIS, 'x', frame_id='MY_CAMERA')
    fov = FlatFOV((4.6e-6, 4.6e-6), (40, 40))     # radians per pixel, pixels
    obs = Snapshot(('u', 'v'), time, 10., fov, 'EARTH', 'MY_CAMERA')

    bp = oops.Backplane(obs)
    bp.ring_radius('SATURN:RING')

A :class:`~oops.frame.TwoVectorFrame` is a frame whose *z*-axis points along one vector
and whose *x*-axis lies toward another; a :class:`~oops.frame.Cmatrix` takes a rotation
matrix instead. The FOV classes model distortion (:class:`~oops.fov.PolynomialFOV`,
:class:`~oops.fov.BarrelFOV`, :class:`~oops.fov.WCSFOV` from a FITS header) and
subarrays; the cadence classes model timing for a :class:`~oops.observation.TimedImage`
whose pixels are exposed at different times. The API reference pages list them all.
