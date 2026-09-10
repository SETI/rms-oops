Examples
========

Each example is a complete script. They use the standard Cassini image
``W1573721822_1``, a wide-angle frame of Saturn, its rings and Epimetheus from November
2007 that ships in the test data tree; substitute any file the host modules read. The
values shown are the ones these scripts print.

Ring radius and longitude of every pixel
----------------------------------------

.. code-block:: python

    import os
    import numpy as np
    import oops
    from oops.hosts.cassini import iss

    path = os.path.join(os.environ['OOPS_RESOURCES'], 'test_data/cassini/ISS/W1573721822_1.IMG')
    obs = iss.from_file(path)
    bp = oops.Backplane(obs, inventory=True)

    radius = bp.ring_radius('SATURN_MAIN_RINGS')
    longitude = bp.ring_longitude('SATURN_MAIN_RINGS', reference='obs') * oops.DPR

    print(radius.shape, radius.mask.mean())
    print(radius.vals[radius.antimask].min(), radius.vals[radius.antimask].max())

    np.save('ring_radius_km.npy', np.where(radius.antimask, radius.vals, np.nan))
    np.save('ring_longitude_deg.npy', np.where(longitude.antimask, longitude.vals, np.nan))

Output::

    (1024, 1024) 0.870
    74659.5 136779.9

The bounded ring body masks every pixel outside the main rings, 87 percent of the
image here; ``'SATURN:RING'`` in place of ``'SATURN_MAIN_RINGS'`` gives the radius in the
unbounded plane at every pixel. The arrays are saved with NaN where masked, which is the
usual way to hand a backplane to other software.

Which pixels are on Saturn, and their lighting
----------------------------------------------

.. code-block:: python

    on_planet = bp.where_intercepted('SATURN')
    incidence = bp.incidence_angle('SATURN') * oops.DPR
    emission = bp.emission_angle('SATURN') * oops.DPR
    phase = bp.phase_angle('SATURN') * oops.DPR

    print(int(on_planet.vals.sum()), 'pixels on Saturn of', on_planet.size)
    print('incidence %.2f to %.2f' % (incidence.vals[incidence.antimask].min(),
                                     incidence.vals[incidence.antimask].max()))
    print('phase %.2f to %.2f' % (phase.vals[phase.antimask].min(),
                                 phase.vals[phase.antimask].max()))

    lit = bp.where_all(('where_intercepted', 'SATURN'),
                       ('where_below', ('incidence_angle', 'SATURN'), np.radians(90.)))
    print(int(lit.vals.sum()), 'lit pixels')

Output::

    314224 pixels on Saturn of 1048576
    incidence 73.26 to 150.34
    phase 60.86 to 62.52
    104821 lit pixels

The lighting backplanes are masked wherever the line of sight misses the planet, so the
antimask selects the disk; ``where_below`` on the incidence angle picks out the day side.
The phase angle barely varies across the disk, because the whole image spans only a few
degrees.

Where a moon is, and its geometry
---------------------------------

.. code-block:: python

    inventory = obs.inventory(['SATURN', 'EPIMETHEUS', 'MIMAS', 'TITAN'], return_type='full')
    for name, info in inventory.items():
        print(name, info['inside'], info['center_uv'], '%.0f km' % info['range'])

    print(bp.center_distance('EPIMETHEUS'))                      # km, a single value
    print(bp.center_phase_angle('EPIMETHEUS') * oops.DPR)
    print(bp.body_diameter_in_pixels('EPIMETHEUS'))
    print(int(bp.where_intercepted('EPIMETHEUS').vals.sum()), 'pixels')

Output::

    SATURN True [-109.56  243.51] 1692966 km
    EPIMETHEUS True [502.24 523.67] 1555030 km
    1 pixels

The inventory reports every body whose disk falls in the field, with the pixel
coordinates of its center even when that center is outside the image, as Saturn's is
here. The gridless ``center_*`` backplanes give a body's geometry as single values
without solving the light path for every pixel.

Sky coordinates and orientation
-------------------------------

.. code-block:: python

    ra = bp.right_ascension() * oops.DPR
    dec = bp.declination() * oops.DPR
    north = bp.celestial_north_angle() * oops.DPR

    print('RA %.3f to %.3f' % (ra.vals.min(), ra.vals.max()))
    print('north angle %.2f' % north.vals.mean())

Output::

    RA 212.273 to 216.471
    north angle 283.36

With no target, the sky backplanes describe each line of sight itself; the celestial
north angle is the direction of north in the image, measured clockwise from up, so a
value near 283 degrees means north points roughly to the left.

A quick look at reduced resolution
----------------------------------

.. code-block:: python

    meshgrid = obs.meshgrid(undersample=8)
    bp8 = oops.Backplane(obs, meshgrid=meshgrid)
    print(bp8.ring_radius('SATURN:RING').shape)

    center = oops.Meshgrid.for_fov_center(obs.fov)
    bp0 = oops.Backplane(obs, meshgrid=center)
    print(bp0.ring_radius('SATURN:RING'))

Output::

    (128, 128)
    Scalar(145136.4245179254)

Undersampling by eight cuts the work by a factor of sixty-four; a meshgrid of the
boresight alone gives the geometry of the center of the field as a single number.

Evaluating a list of backplanes
-------------------------------

.. code-block:: python

    keys = [('ring_radius', 'SATURN:RING'),
            ('ring_longitude', 'SATURN:RING', 'node'),
            ('phase_angle', 'SATURN'),
            ('longitude', 'SATURN', 'iau', 'west', 0, 'graphic'),
            ('latitude', 'SATURN', 'graphic'),
            ('where_in_front', 'SATURN:RING', 'SATURN')]

    for key in keys:
        array = bp.evaluate(key)
        valid = array.vals[array.antimask]
        print(key, array.shape, valid.min(), valid.max())

Every backplane method has a key form, the method name followed by its arguments in
order, which is how a configuration file or a table can name the backplanes to produce.
``('where_in_front', 'SATURN:RING', 'SATURN')`` is the ring plane where the planet does
not hide it.

Reading backplanes written by the gold master tool
--------------------------------------------------

After ``python tests/hosts/cassini/iss/gold_master.py --preview -o /tmp/preview``:

.. code-block:: python

    import ast
    import pickle

    root = '/tmp/preview/cassini.iss/W1573721822_1'
    with open(f'{root}/arrays/SATURN-RING radius (km).pickle', 'rb') as f:
        radius = pickle.load(f)
    print(radius.shape, radius.vals[radius.antimask].max())

    with open(f'{root}/summary.py') as f:
        text = ''.join(line for line in f if not line.startswith('#'))
    summary = ast.literal_eval(text)
    print(summary['SATURN incidence angle, actual (deg)'])

The pickles are ``polymath`` arrays, so ``vals``, ``mask`` and ``antimask`` work as they
do on a freshly computed backplane; the summary is the range of every backplane in one
dictionary. On Linux the file names have underscores in place of spaces
(:doc:`user_guide_gold_master`).
