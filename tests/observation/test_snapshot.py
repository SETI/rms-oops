##########################################################################################
# tests/observation/test_snapshot.py
##########################################################################################

import numpy as np
import pytest

from polymath         import Pair, Vector
from oops.body        import Body
from oops.fov         import FlatFOV
from oops.frame       import Cmatrix, Frame
from oops.observation import Snapshot
from oops.path        import Path


def test_snapshot():
    fov = FlatFOV((0.001,0.001), (10,20))
    obs = Snapshot(('u','v'), tstart=98., texp=2.,
                   fov=fov, path='SSB', frame='J2000')

    indices = Vector([(0.,0.),(0.,20.),(10.,0.),(10.,20.),(10.,21.)])
    indices_ = indices.copy()
    indices_.vals[:,0][indices.vals[:,0] == 10] -= 1
    indices_.vals[:,1][indices.vals[:,1] == 20] -= 1

    # uvt() with remask == False
    (uv,time) = obs.uvt(indices)

    assert not uv.mask
    assert not time.mask
    assert time == 99.
    assert uv == Pair.as_pair(indices)

    # uvt() with remask == True
    (uv,time) = obs.uvt(indices, remask=True)

    assert np.all(uv.mask == np.array(4*[False] + [True]))
    assert np.all(time.mask == uv.mask)
    assert time[:4] == 99.
    assert uv[:4] == Pair.as_pair(indices)[:4]

    # uvt_range() with remask == False
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices)

    assert not uv_min.mask
    assert not uv_max.mask
    assert not time_min.mask
    assert not time_max.mask

    assert uv_min == Pair.as_pair(indices_)
    assert uv_max == Pair.as_pair(indices_) + (1,1)
    assert time_min == 98.
    assert time_max == 100.

    # uvt_range() with remask == False, new indices
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9))

    assert not uv_min.mask
    assert not uv_max.mask
    assert not time_min.mask
    assert not time_max.mask

    assert uv_min == Pair.as_pair(indices)
    assert uv_max == Pair.as_pair(indices) + (1,1)
    assert time_min == 98.
    assert time_max == 100.

    # uvt_range() with remask == True, new indices
    (uv_min, uv_max, time_min, time_max) = obs.uvt_range(indices+(0.2,0.9),
                                                         remask=True)
    assert np.all(uv_min.mask == [False] + 4*[True])
    assert np.all(uv_min.mask == uv_max.mask)
    assert np.all(uv_min.mask == time_min.mask)
    assert np.all(uv_min.mask == time_max.mask)

    assert uv_min[0] == Pair.as_pair(indices)[0]
    assert uv_max[0] == (Pair.as_pair(indices) + (1,1))[0]
    assert time_min[0] == 98.
    assert time_max[0] == 100.

    # time_range_at_uv() with remask == False
    uv_pair = Pair([(0.,0.),(0.,20.),(10.,0.),(10.,20.),(10.,21.)])

    (time0, time1) = obs.time_range_at_uv(uv_pair)

    assert time0 == 98.
    assert time1 == 100.

    # time_range_at_uv() with remask == True
    (time0, time1) = obs.time_range_at_uv(uv_pair, remask=True)

    assert np.all(time0.mask == 4*[False] + [True])
    assert np.all(time1.mask == 4*[False] + [True])
    assert time0[:4] == 98.
    assert time1[:4] == 100.

    # Alternative axis order ('v','u')
    obs = Snapshot(('v','u'), tstart=98., texp=2.,
                   fov=fov, path='SSB', frame='J2000')

    indices = Pair([(0,0),(0,10),(20,0),(20,10),(20,11)])

    (uv,time) = obs.uvt(indices)

    assert uv == indices.to_pair((1,0))

    (uv,time) = obs.uvt(indices, remask=True)

    assert uv[:4] == indices.to_pair((1,0))[:4]
    assert np.all(uv.mask == 4*[False] + [True])

    # Alternative axis order ('v', 'a', 'u')
    obs = Snapshot(('v','a','u'), tstart=98., texp=2.,
                   fov=fov, path='SSB', frame='J2000')

    indices = Vector([(0,-1,0),(0,99,10),(20,-9,0),(20,77,10),(20,44,11)])
    (uv,time) = obs.uvt(indices)

    assert uv == indices.to_pair((2,0))

    (uv,time) = obs.uvt(indices, remask=True)

    assert uv[:4] == indices.to_pair((2,0))[:4]
    assert np.all(uv.mask == 4*[False] + [True])


@pytest.fixture(scope='module', autouse=True)
def _solar_system():
    Body.reset_registry()
    Body._undefine_solar_system()
    Body.define_solar_system('2000-01-01', '2010-01-01')
    yield
    Frame._reset_caches()
    Path._reset_caches()
    Body.reset_registry()


def _snapshot_facing_the_moon() -> Snapshot:
    """A snapshot from Earth whose FOV is centered on the Moon at time zero.

    Returns:
        Snapshot: A 100 by 100 observation in which the Moon is inside the FOV and the
        other planets are not.
    """

    frame = Cmatrix.from_ra_dec(222.447, -10.900, 0., 'J2000')
    return Snapshot(('u','v'), 0., 10., FlatFOV((1.e-3, 1.e-3), (100, 100)),
                    'EARTH', frame)


# The keys of a "full" inventory entry, with the types the docstring documents.
INVENTORY_KEYS = {
    'name':             str,
    'inside':           (bool, np.bool_),
    'center_uv':        np.ndarray,
    'center':           np.ndarray,
    'range':            (float, np.floating),
    'outer_radius':     (float, np.floating),
    'inner_radius':     (float, np.floating),
    'resolution':       np.ndarray,
    'u_min':            (int, np.integer),
    'u_max':            (int, np.integer),
    'v_min':            (int, np.integer),
    'v_max':            (int, np.integer),
    'u_min_unclipped':  (int, np.integer),
    'u_max_unclipped':  (int, np.integer),
    'v_min_unclipped':  (int, np.integer),
    'v_max_unclipped':  (int, np.integer),
    'u_pixel_size':     (float, np.floating),
    'v_pixel_size':     (float, np.floating),
}

BODIES = ['MOON', 'MARS', 'JUPITER']


def test_inventory_full_returns_an_entry_for_every_body() -> None:
    """A "full" inventory describes every body asked about, inside the FOV or not."""

    full = _snapshot_facing_the_moon().inventory(BODIES, return_type='full')

    assert sorted(full) == sorted(BODIES)


def test_inventory_full_flags_which_bodies_are_inside() -> None:
    """The "inside" value carries what the "list" and "flags" return types select."""

    obs = _snapshot_facing_the_moon()
    full = obs.inventory(BODIES, return_type='full')

    assert [bool(full[name]['inside']) for name in BODIES] == [True, False, False]
    assert obs.inventory(BODIES, return_type='list') == ['MOON']


def test_inventory_full_agrees_with_the_other_return_types() -> None:
    """One body is inside here, so the three return types must say so consistently."""

    obs = _snapshot_facing_the_moon()
    full = obs.inventory(BODIES, return_type='full')
    flags = obs.inventory(BODIES, return_type='flags')
    names = obs.inventory(BODIES, return_type='list')

    for (i, name) in enumerate(BODIES):
        assert bool(full[name]['inside']) == bool(flags[i])
        assert bool(full[name]['inside']) == (name in names)


@pytest.mark.parametrize('key,expected', sorted(INVENTORY_KEYS.items()))
def test_inventory_full_entry_matches_its_documented_type(key: str, expected) -> None:
    """Each documented key is present and holds a value of its documented type."""

    entry = _snapshot_facing_the_moon().inventory(BODIES, return_type='full')['MOON']

    assert key in entry
    assert isinstance(entry[key], expected)


def test_inventory_full_entry_has_no_undocumented_keys() -> None:
    """The entry holds exactly the keys the docstring lists, and no others."""

    entry = _snapshot_facing_the_moon().inventory(BODIES, return_type='full')['MOON']

    assert sorted(entry) == sorted(INVENTORY_KEYS)


def test_inventory_full_pair_valued_entries_have_the_documented_widths() -> None:
    """The array-valued entries hold two floats, except the three-component center."""

    entry = _snapshot_facing_the_moon().inventory(BODIES, return_type='full')['MOON']

    assert entry['center_uv'].shape == (2,)
    assert entry['resolution'].shape == (2,)
    assert entry['center'].shape == (3,)

##########################################################################################
