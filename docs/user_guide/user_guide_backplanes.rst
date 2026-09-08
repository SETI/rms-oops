Backplanes
==========

Computing a backplane
---------------------

A :class:`~oops.Backplane` wraps an observation and computes geometric arrays over it on
request:

.. code-block:: python

    import oops
    from oops.hosts.cassini import iss

    obs = iss.from_file('/path/to/W1573721822_1.IMG')
    bp = oops.Backplane(obs, inventory=True)

    radius = bp.ring_radius('SATURN:RING')          # km, at every pixel
    incidence = bp.incidence_angle('SATURN')        # radians, masked off the planet
    on_planet = bp.where_intercepted('SATURN')      # Boolean

Each method returns a ``polymath`` array with the shape of the meshgrid, here the
(1024, 1024) of the image, masked where the quantity is undefined. Nothing is computed
until asked for, and everything computed is cached: the first backplane about a surface
solves the light path from every pixel to that surface, which takes a few seconds for a
full-resolution Cassini image, and later backplanes about the same surface reuse the
solution. Asking twice for the same backplane returns the same object.

The constructor's options:

``meshgrid``
    Where to evaluate; default is the center of every pixel. Pass an undersampled
    meshgrid (:doc:`user_guide_observations`) for a quick look.
``time``
    The time of each sample; default is the midpoint of each pixel's exposure.
``inventory``
    ``True`` to find which bodies are in view first and solve the light path only
    inside each body's bounding box, which is much faster when a body covers a small
    part of the image. ``inventory_border`` widens the boxes by a number of pixels, for
    a wide or distorted field where the spherical approximation is rough; the Galileo
    and JunoCam host tests turn the inventory off entirely. Default ``None``.

Naming a target
---------------

The first argument of nearly every backplane method is an *event key*, which says which
surface the photon left and, implicitly, that the Sun illuminated it. In its simplest
form it is a body name, in any case:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``'SATURN'``
     - The body's own surface, a spheroid or ellipsoid
   * - ``'SATURN:RING'``
     - The unbounded ring plane of the body
   * - ``'SATURN:ANSA'``
     - The ansa: the ring plane seen edge-on at its extremities
   * - ``'SATURN:LIMB'``
     - The limb: the atmosphere's edge as seen from the observer
   * - ``'SATURN_MAIN_RINGS'``
     - A registered ring body, whose ring plane is bounded in radius; ``'SATURN_MAIN_RINGS:ANSA'`` is its ansa
   * - ``'EPIMETHEUS'``
     - Any moon, likewise with ``:LIMB``

The registered ring bodies are ``SATURN_MAIN_RINGS``, ``SATURN_A_RING``,
``SATURN_B_RING``, ``SATURN_C_RING``, ``SATURN_AB_RINGS``, ``SATURN_RINGS`` and
``SATURN_RING_SYSTEM``; ``JUPITER_RING_SYSTEM``; ``URANUS_RING_SYSTEM``, ``MU_RING``,
``NU_RING`` and the ten named narrow rings (``EPSILON_RING``, ``ALPHA_RING``, ...);
``NEPTUNE_RING_SYSTEM``; and a ``<PLANET>_RING_PLANE`` for each. ``Body.exists(name)``
checks a name. Ring radii outside a bounded ring's limits are masked, which is the
difference between ``'SATURN:RING'``, valid at every pixel, and
``'SATURN_MAIN_RINGS'``, valid only where the main rings are.

The full form of an event key is a tuple ``(source, surface)``, and a bare name is
shorthand for ``('SUN<', name)``: illumination from the Sun, dispersed off the surface
toward the observer. Two other sources appear in special cases. ``('SUN>', surface)``
is an occultation: the source seen through the surface, as in a solar occultation by the
rings. ``('SUN-', surface)`` is *gridless*: the photon from the body's center rather
than from each pixel, which gives a single value rather than an array. The methods
named ``center_*`` and ``sub_*`` form the gridless key themselves, so
``bp.center_phase_angle('SATURN')`` is the phase angle at Saturn's center and
``bp.sub_observer_latitude('SATURN')`` the latitude beneath the observer. A third item
names a shadowing surface: ``('SUN<', 'MIMAS', 'SATURN:RING')`` is the surface of Mimas
in the shadow of the rings, which the ``where_inside_shadow`` method builds for you.

Keys as arguments
-----------------

Some methods take a *backplane key* instead of an event key: the ``where_*`` and
``border_*`` families, which derive a mask from another backplane. A backplane key is
the method name followed by its arguments, and a backplane array already computed can
be passed in its place:

.. code-block:: python

    inner = bp.where_below(('ring_radius', 'SATURN:RING'), 100000.)
    lit = bp.where_all(('where_intercepted', 'SATURN:RING'),
                       ('where_sunward', 'SATURN:RING'))
    edge = bp.border_inside(bp.where_intercepted('SATURN'))

``bp.evaluate(key)`` computes any backplane from its key, which is how a list of
backplanes read from a file or a table can be produced in a loop:

.. code-block:: python

    for key in [('ring_radius', 'SATURN:RING'), ('phase_angle', 'SATURN'),
                ('longitude', 'SATURN', 'iau', 'west', 0, 'graphic')]:
        array = bp.evaluate(key)

Every registered array carries its own key in the attribute ``key``.

Using the results
-----------------

A result is a :class:`~oops.Scalar` or :class:`~oops.Boolean` with ``vals``, ``mask``
and ``antimask``. Take the valid values with the antimask; convert angles with
``oops.DPR``; combine masks with the ``where_*`` methods or with NumPy:

.. code-block:: python

    >>> radius = bp.ring_radius('SATURN_MAIN_RINGS')
    >>> radius.shape, radius.mask.mean()
    ((1024, 1024), 0.87)
    >>> radius.vals[radius.antimask].min(), radius.vals[radius.antimask].max()
    (74658.6, 136779.9)
    >>> phase = bp.phase_angle('SATURN') * oops.DPR
    >>> lit_ring = bp.where_sunward('SATURN:RING').vals & ~radius.mask

Registered arrays are read-only and shared with every later caller of the same key.
Do not modify one in place; arithmetic and NumPy indexing return new objects.

The catalog
-----------

The methods are grouped by family. Every event key is written ``key`` below; the
optional arguments are shown with their defaults. Angles are radians in the arrays,
whatever the name says. ``apparent`` selects the aberrated direction (``True``) or the
actual one.

Sky
~~~

The observation itself is the surface here: with no key, or the empty tuple, the value
is that of each line of sight.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``right_ascension(key=(), apparent=True, direction='arr')``
     - Right ascension of the photon.
   * - ``declination(key=(), apparent=True, direction='arr')``
     - Declination of the photon.
   * - ``celestial_north_angle(key=())``
     - Direction of celestial north at each pixel, clockwise from up.
   * - ``celestial_east_angle(key=())``
     - Direction of celestial east.
   * - ``center_right_ascension(key, apparent=True, direction='arr')``
     - Gridless right ascension of the body center.
   * - ``center_declination(key, apparent=True, direction='arr')``
     - Gridless declination of the body center.

Distance and time
~~~~~~~~~~~~~~~~~

``direction`` is ``'dep'`` for the photon departing the surface toward the observer or
``'arr'`` for the one arriving from the Sun.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``distance(key, direction='dep')``
     - Distance in km the photon traveled.
   * - ``light_time(key, direction='dep')``
     - The same as a time in seconds.
   * - ``event_time(key)``
     - Time in seconds TDB when the photon left the surface.
   * - ``center_distance(key, direction='dep')``
     - Gridless distance to the body center.
   * - ``center_light_time(key, direction='dep')``
     - Gridless light time.
   * - ``center_time(key)``
     - Gridless event time.

Lighting
~~~~~~~~

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``incidence_angle(key, apparent=True)``
     - Angle between the surface normal and the Sun.
   * - ``emission_angle(key, apparent=True)``
     - Angle between the normal and the observer.
   * - ``phase_angle(key, apparent=True)``
     - Angle at the surface between the Sun and the observer.
   * - ``scattering_angle(key, apparent=True)``
     - 180 degrees minus the phase angle.
   * - ``mu0(key, apparent=True)``
     - Cosine of the incidence angle.
   * - ``mu(key, apparent=True)``
     - Cosine of the emission angle.
   * - ``lambert_law(key)``
     - A Lambert reflectance model.
   * - ``lommel_seeliger_law(key)``
     - A Lommel-Seeliger model.
   * - ``minnaert_law(key, k, k2=None, clip=0.2)``
     - A Minnaert model with exponent ``k``.
   * - ``center_incidence_angle(key, apparent=True)``
     - Gridless incidence angle at the body center.
   * - ``center_emission_angle(key, apparent=True)``
     - Gridless emission angle.
   * - ``center_phase_angle(key, apparent=True)``
     - Gridless phase angle.
   * - ``center_scattering_angle(key, apparent=True)``
     - Gridless scattering angle.

Body surfaces
~~~~~~~~~~~~~

For spheroids and ellipsoids. ``reference`` is ``'iau'`` (the body's prime meridian),
``'sun'``, ``'sha'`` (the sub-solar longitude), ``'obs'`` or ``'oha'`` (the sub-observer
longitude); ``direction`` is ``'west'`` or ``'east'``; ``minimum`` is 0 or -180 for the
range of the result; ``lon_type`` and ``lat_type`` are ``'centric'``, ``'graphic'`` or
``'squashed'``.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``longitude(key, reference='iau', direction='west', minimum=0, lon_type='centric')``
     - Longitude at the intercept.
   * - ``latitude(key, lat_type='centric')``
     - Latitude at the intercept.
   * - ``sub_observer_longitude(key, reference='iau', direction='west', minimum=0)``
     - Gridless sub-observer longitude.
   * - ``sub_solar_longitude(key, reference='iau', direction='west', minimum=0)``
     - Gridless sub-solar longitude.
   * - ``sub_observer_latitude(key, lat_type='centric')``
     - Gridless sub-observer latitude.
   * - ``sub_solar_latitude(key, lat_type='centric')``
     - Gridless sub-solar latitude.

Rings
~~~~~

For ``:RING`` keys and ring bodies. ``reference`` for a longitude is ``'node'`` (the
ascending node on J2000), ``'aries'``, ``'sun'``, ``'sha'``, ``'obs'`` or ``'oha'``;
``pole`` is ``'sunward'``, ``'north'``, ``'prograde'`` or ``'unsigned'``, choosing the
convention for angles measured from the ring plane's normal.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``ring_radius(key, rmin=None, rmax=None)``
     - Radius in the ring plane, km, masked outside the limits.
   * - ``ring_longitude(key, reference='node')``
     - Longitude in the ring plane.
   * - ``ring_azimuth(key, direction='obs', apparent=True)``
     - Angle from the local radial direction to the observer or Sun.
   * - ``ring_elevation(key, direction='obs', pole='prograde', apparent=True)``
     - Angle of the observer or Sun above the ring plane.
   * - ``ring_incidence_angle(key, pole='sunward', apparent=True)``
     - Incidence angle under the pole convention.
   * - ``ring_emission_angle(key, pole='sunward', apparent=True)``
     - Emission angle under the pole convention.
   * - ``ring_radial_resolution(key)``
     - Radial resolution, km per pixel.
   * - ``ring_angular_resolution(key, units='rad')``
     - Angular resolution per pixel.
   * - ``ring_gradient_angle(key)``
     - Direction of the radius gradient in the image.
   * - ``radial_mode(backplane_key, cycles, epoch, amp, peri0, speed, a0=0., dperi_da=0., reference='node')``
     - A ring radius shifted by a normal-mode perturbation.
   * - ``ring_sub_observer_longitude(key, reference='node')``
     - Gridless sub-observer longitude.
   * - ``ring_sub_solar_longitude(key, reference='node')``
     - Gridless sub-solar longitude.
   * - ``ring_center_incidence_angle(key, pole='sunward', apparent=True)``
     - Incidence angle at the ring center.
   * - ``ring_center_emission_angle(key, pole='sunward', apparent=True)``
     - Emission angle at the ring center.
   * - ``ring_shadow_radius(key, ring_surface_key)``
     - Radius in the ring plane whose shadow falls on each point of the body.
   * - ``ring_shadow_incidence(key, ring_surface_key)``
     - Incidence angle in the ring plane at that radius.
   * - ``ring_radius_in_front(key, ring_surface_key)``
     - Radius in the ring plane that obscures each point of the body.

Ansas
~~~~~

For ``:ANSA`` keys: the rings seen edge-on at their projected extremities.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``ansa_radius(key, radius_type='positive', rmax=None)``
     - Radius of the ansa intercept, km.
   * - ``ansa_altitude(key)``
     - Height above the ring plane, km.
   * - ``ansa_longitude(key, reference='node')``
     - Longitude of the intercept.
   * - ``ansa_radial_resolution(key)``
     - Radial resolution, km per pixel.
   * - ``ansa_vertical_resolution(key)``
     - Vertical resolution.

Limbs
~~~~~

For ``:LIMB`` keys: the atmosphere along the edge of the body.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``limb_altitude(key, zmin=None, zmax=None, scaled=False)``
     - Height of the limb point above the surface, km.
   * - ``limb_longitude(key, reference='iau', direction='west', minimum=0, lon_type='centric')``
     - Longitude of the limb point.
   * - ``limb_latitude(key, lat_type='centric')``
     - Latitude of the limb point.
   * - ``limb_clock_angle(key)``
     - Angle around the limb, clockwise from the projected pole.

Resolution, pole, orbit and pixel geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``resolution(key, axis='u')``
     - Km per pixel along an image axis at the intercept.
   * - ``finest_resolution(key)``
     - Km per pixel in the best direction.
   * - ``coarsest_resolution(key)``
     - Km per pixel in the worst direction.
   * - ``center_resolution(key, axis='u')``
     - Gridless resolution at the body center.
   * - ``pole_clock_angle(key)``
     - Gridless clock angle of the projected pole.
   * - ``pole_position_angle(key)``
     - Gridless position angle of the projected pole.
   * - ``orbit_longitude(key, reference='obs', planet=None)``
     - Gridless longitude of a moon in its orbit about its planet.
   * - ``body_diameter_in_pixels(key, radius=0, axis='max')``
     - Gridless apparent diameter of the body in pixels.
   * - ``center_coordinate(key, axis='u')``
     - Gridless *u* or *v* of the body center.

Masks
~~~~~

Boolean backplanes. ``tvl`` selects three-valued logic, in which a masked pixel stays
masked rather than becoming False.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``where_intercepted(key)``
     - True where the line of sight meets the surface.
   * - ``where_sunward(key, tvl=False)``
     - True where the surface faces the Sun.
   * - ``where_antisunward(key, tvl=False)``
     - True where it faces away.
   * - ``where_in_front(key, surface_key, tvl=False)``
     - True where the surface is not hidden by a second surface.
   * - ``where_in_back(key, surface_key, tvl=False)``
     - True where it is hidden.
   * - ``where_inside_shadow(key, surface_key, tvl=False)``
     - True where a second body shades the surface.
   * - ``where_outside_shadow(key, surface_key, tvl=False)``
     - True where it does not.
   * - ``where_inside(key, surface_key, tvl=False)``
     - True where the surface is inside a second one.
   * - ``where_outside(key, surface_key, tvl=False)``
     - True where it is outside.
   * - ``where_below(backplane_key, value, tvl=False)``
     - True where a backplane is at or below a value.
   * - ``where_above(backplane_key, value, tvl=False)``
     - True where it is at or above.
   * - ``where_between(backplane_key, low, high, tvl=False)``
     - True where it is in a range.
   * - ``where_not(backplane_key, tvl=False)``
     - The logical negation.
   * - ``where_any(*backplane_keys, tvl=False)``
     - True where any is True.
   * - ``where_all(*backplane_keys, tvl=False)``
     - True where all are True.

Borders
~~~~~~~

The pixels along an edge, as Boolean backplanes.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Method
     - Meaning
   * - ``border_above(backplane_key, value)``
     - Pixels at or above a value next to one below it.
   * - ``border_below(backplane_key, value)``
     - The converse.
   * - ``border_atop(backplane_key, value)``
     - The pixels straddling the contour.
   * - ``border_inside(backplane_key)``
     - True pixels of a mask next to a False one.
   * - ``border_outside(backplane_key)``
     - False pixels next to a True one.
