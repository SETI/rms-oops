##########################################################################################
# tests/test_transform.py
##########################################################################################

import numpy as np
import pytest

from polymath   import Scalar, Vector3, Matrix3
from oops       import Transform
from oops.frame import Frame, Rotation
from oops.path  import Path


def test_transform():
    np.random.seed(5819)

    # Fake out the FRAME REGISTRY with something that has .shape = ()
    Frame._FRAME_REGISTRY["TEST"] = Frame.as_wayframe("J2000")
    Frame._FRAME_REGISTRY["SPIN"] = Frame.as_wayframe("J2000")

    tr = Transform(Matrix3(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])),
                   Vector3(np.array([0.,0.,0.])), "J2000", "J2000")

    p = Vector3(np.random.rand(2,1,4,3))
    v = Vector3(np.random.rand(  3,4,3))

    assert tr.rotate_pos_vel(p,v)[0] == p
    assert tr.rotate_pos_vel(p,v)[1] == v

    assert tr.unrotate_pos_vel(p,v)[0] == p
    assert tr.unrotate_pos_vel(p,v)[1] == v

    tr = tr.invert()

    assert tr.rotate_pos_vel(p,v)[0] == p
    assert tr.rotate_pos_vel(p,v)[1] == v

    assert tr.unrotate_pos_vel(p,v)[0] == p
    assert tr.unrotate_pos_vel(p,v)[1] == v

    tr = Transform(Matrix3([[1,0,0],[0,1,0],[0,0,1]]),
                   Vector3([0,0,1]), "SPIN", "J2000")

    assert tr.unrotate_pos_vel(p,v)[0] == p
    assert Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,2]) == Scalar(v.mvals[...,2])
    assert (Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,0])
            == Scalar(v.mvals[...,0]) + Scalar(p.mvals[...,1]))
    assert (Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,1])
            == Scalar(v.mvals[...,1]) - Scalar(p.mvals[...,0]))

    tr = tr.invert()

    assert tr.rotate(p) == p
    assert Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,2]) == Scalar(v.mvals[...,2])
    assert (Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,0])
            == Scalar(v.mvals[...,0]) - Scalar(p.mvals[...,1]))
    assert (Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,1])
            == Scalar(v.mvals[...,1]) + Scalar(p.mvals[...,0]))

    a = Vector3(np.random.rand(3,1,3))
    b = Vector3(np.random.rand(1,1,3))
    m = Matrix3.twovec(a,0,b,1)
    omega = Vector3(np.random.rand(3,1,3))

    tr = Transform(m, omega, "TEST", "J2000")

#         self.assertEqual(tr.unrotate(p), tr.invert().rotate(p))
#         self.assertEqual(tr.rotate(p), tr.invert().unrotate(p))
    eps = 1.e-15
    assert np.all(np.abs(tr.unrotate(p).vals - tr.invert().rotate(p).vals)) < eps
    assert np.all(np.abs(tr.rotate(p).vals - tr.invert().unrotate(p).vals)) < eps

    eps = 1.e-15
    diff = tr.unrotate_pos_vel(p,v)[1] - tr.invert().rotate_pos_vel(p,v)[1]
    assert np.all(diff.vals > -eps)
    assert np.all(diff.vals <  eps)

    diff = tr.rotate_pos_vel(p,v)[1] - tr.invert().unrotate_pos_vel(p,v)[1]
    assert np.all(diff.vals > -eps)
    assert np.all(diff.vals <  eps)

        # Transform derivatives are unit tested as part of the SpinFrame tests

##########################################################################################
# The origin, the description, and the transform-on-transform operations
##########################################################################################

# A quarter-turn about the z-axis, and a rotation rate about the same axis
QUARTER_TURN = Matrix3([(0., 1., 0.), (-1., 0., 0.), (0., 0., 1.)])
SPIN = Vector3((0., 0., 1.e-4))


def test_the_origin_can_be_named_by_its_path_id(core_kernels) -> None:
    """A string origin is looked up in the path registry."""

    from oops.path import SpicePath

    SpicePath('MARS', 'SSB')
    xform = Transform(Matrix3.IDENTITY, Vector3.ZERO, 'J2000', 'J2000', origin='MARS')

    assert xform.origin is Path.as_waypoint('MARS')


def test_the_origin_can_be_given_as_a_path(core_kernels) -> None:
    """A Path is reduced to its waypoint."""

    xform = Transform(Matrix3.IDENTITY, Vector3.ZERO, 'J2000', 'J2000',
                      origin=Path.SSB)

    assert xform.origin is Path.SSB.waypoint


def test_the_description_names_the_shape_and_the_frame() -> None:
    """A Transform prints its shape and the frame it rotates into."""

    xform = Transform(Matrix3.IDENTITY, Vector3.ZERO, 'J2000', 'J2000')

    assert str(xform) == "Transform(shape=()/'J2000')"


def test_rotate_accepts_a_bare_triple() -> None:
    """Anything that is not a PolyMath object is converted to a Vector3 first."""

    xform = Transform(QUARTER_TURN, Vector3.ZERO, 'J2000', 'J2000')

    assert xform.rotate((1., 0., 0.)) == xform.rotate(Vector3((1., 0., 0.)))


def test_unrotate_accepts_a_bare_triple() -> None:
    """The same holds in the reverse direction."""

    xform = Transform(QUARTER_TURN, Vector3.ZERO, 'J2000', 'J2000')

    assert xform.unrotate((1., 0., 0.)) == xform.unrotate(Vector3((1., 0., 0.)))


def test_rotate_transform_requires_matching_frames() -> None:
    """The reference frame of this Transform must be the target frame of the other."""

    other = Rotation(0.25, 2, Frame.J2000, frame_id='TEST_TRANSFORM_OTHER')
    outer = Transform(QUARTER_TURN, SPIN, 'J2000', other)
    inner = Transform(Matrix3.IDENTITY, Vector3.ZERO, 'J2000', 'J2000')

    with pytest.raises(ValueError, match='frame mismatch in rotate_transform'):
        outer.rotate_transform(inner)


def test_unrotate_transform_inverts_rotate_transform() -> None:
    """Applying the inverse of a Transform undoes what applying it does."""

    first = Transform(QUARTER_TURN, SPIN, 'J2000', 'J2000')
    second = Transform(QUARTER_TURN, Vector3.ZERO, 'J2000', 'J2000')

    combined = second.rotate_transform(first)
    recovered = second.unrotate_transform(combined)

    assert np.abs(recovered.matrix.vals - first.matrix.vals).max() < 1.e-14
    assert np.abs(recovered.omega.vals - first.omega.vals).max() < 1.e-14

##########################################################################################
