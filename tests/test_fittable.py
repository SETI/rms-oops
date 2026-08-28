##########################################################################################
# tests/test_fittable.py
##########################################################################################

import pytest

from oops.fittable import Fittable
import oops.mutable as mutable


class A(Fittable):
    nparams = 1

    def __init__(self, x):
        self.x = x
        self._refresh()

    def _refresh(self):
        self.x_squared = self.x**2

    def _set_params(self, params):
        self.x = params[0]

    @property
    def params(self):
        return (self.x,)

class B:
    def __init__(self, x, a):
        self.x = x
        self.a = a
        self._refresh()

    def _refresh(self):
        self.x_plus_a2 = self.x + self.a.x_squared


class C(Fittable):
    nparams = 1

    def __init__(self, x, a):
        self.x = x
        self.a = a
        self.c = self
        self._refresh()

    def _set_params(self, params):
        self.x = params[0]

    @property
    def params(self):
        return (self.x,)

    def _refresh(self):
        self.x_plus_a2_plus_cx_plus_ccx = (self.x + self.a.x_squared + self.c.x
                                           + self.c.c.x)


class D:
    def __init__(self, x):
        self.x = x


def test_fittable():
    x = ()
    assert not mutable.is_fittable(x)
    assert mutable.mutable_names(x) == []
    assert mutable.version(x) == 0

    a = A(7)
    assert a.x_squared == 49
    assert a.params == (7,)
    assert a.version == 0
    assert isinstance(a.params, tuple)
    assert mutable.is_fittable(a)
    assert mutable.is_mutable(a)
    assert mutable.mutable_names(a) == []
    assert mutable.version(a) == 0

    a.set_params([5])
    assert a.params == (5,)
    assert a.x_squared == 25
    assert a.version > 0
    assert mutable.mutable_names(a) == []
    assert mutable.is_fittable(a)

    b = B(1, a)
    mutable.set_param_order(b, 'a')
    assert b.x_plus_a2 == 26
    assert mutable.get_params(b) == (5.,)
    assert isinstance(mutable.get_params(b)[0], float)
    assert not mutable.is_fittable(b)
    assert mutable.is_mutable(b)

    mutable.set_params(b, 7)
    assert b.x_plus_a2 == 50
    assert mutable.mutable_names(b) == ['a']
    assert mutable.unfrozen_names(b) == ['a']

    mutable.freeze(a)
    assert mutable.mutable_names(b) == ['a']
    assert mutable.unfrozen_names(b) == []

    a = A(5)
    c = C(1, a)
    assert mutable.mutable_names(c) == ['a']
    assert mutable.is_fittable(c)
    assert c.x_plus_a2_plus_cx_plus_ccx == 28

    a.set_params([6])
    assert c.refresh()
    assert c.x_plus_a2_plus_cx_plus_ccx == 39
    assert not c.refresh()
    assert not c.refresh()

    a = A(5)
    c = C(1, a)
    assert c.x_plus_a2_plus_cx_plus_ccx == 28

    mutable.set_param_order(c, ['', 'a'])
    assert mutable.get_param_order(c) == ['', 'a']
    assert mutable.get_nparams(c) == 2
    assert mutable.get_params(c) == (1,5)

    mutable.set_params(c, [1, 6])
    assert c.x_plus_a2_plus_cx_plus_ccx == 39
    assert not c.refresh()
    assert mutable.get_params(a) == (6,)
    assert mutable.get_params(c) == (1,6)

    assert not mutable.is_frozen(a)
    assert not mutable.is_frozen(c)
    assert not a.is_frozen
    assert not c.is_frozen

    mutable.freeze(a)
    assert mutable.is_frozen(a)
    assert not mutable.is_frozen(c)
    with pytest.raises(ValueError):
        mutable.set_params(a, 2)

    assert mutable.mutable_names(c) == ['a']
    assert mutable.unfrozen_names(c) == []
    assert mutable.is_fittable(a)
    assert mutable.is_fittable(c)
    assert mutable.get_params(c) == (1,6)
    assert c.params == (1,)

    a = A(5)
    c = C(1, a)
    mutable.freeze(c)
    assert mutable.is_frozen(a)
    assert mutable.is_frozen(c)
    assert a.is_frozen
    assert c.is_frozen
    with pytest.raises(ValueError):
        mutable.set_params(a, 2)
    with pytest.raises(ValueError):
        mutable.set_params(c, 2)

    assert mutable.unfrozen_names(c) == []
    assert mutable.mutable_names(c) == ['a']
    assert mutable.is_fittable(a)
    assert mutable.is_fittable(c)

    # type `class` has __data__ but is immutable
    d = D(int)
    assert len(mutable._IMMUTABLE_OBJECTS) == 0
    assert mutable.get_params(d) == ()
    assert mutable.is_frozen(d)
    assert len(mutable._IMMUTABLE_OBJECTS) == 1
    assert not mutable.set_params(d, ())
    with pytest.raises(ValueError):
        mutable.set_params(d, (1.,))

    d = D(float)
    assert mutable.freeze(d) is False  # tests TypeError check in freeze()
##########################################################################################
