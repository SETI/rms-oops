##########################################################################################
# tests/test_mutable.py
##########################################################################################

import pytest

from oops          import mutable
from oops.fov      import FlatFOV, OffsetFOV, SliceFOV

def _flat() -> FlatFOV:
    """A plain FlatFOV, which carries no fittable parameters.

    A fresh one is built for each test: freezing or fitting an object that wraps it
    reaches the FlatFOV too, so a shared instance would leak between tests.
    """

    return FlatFOV((1.e-4, 1.2e-4), (60, 40))


def _fittable() -> OffsetFOV:
    """An OffsetFOV, which is Fittable through its (u,v) offset."""

    return OffsetFOV(_flat(), uv_offset=(1., 2.))


def _holder() -> SliceFOV:
    """A SliceFOV wrapping a Fittable OffsetFOV, so it is mutable but not Fittable."""

    return SliceFOV(_fittable(), (0, 0), (10, 10))


def test_is_fittable_of_a_fittable_object() -> None:
    """An object that subclasses Fittable is Fittable."""

    assert mutable.is_fittable(_fittable())


def test_is_fittable_of_a_holder() -> None:
    """An object that merely contains a Fittable is not itself Fittable."""

    assert not mutable.is_fittable(_holder())


def test_is_fittable_of_an_immutable_object() -> None:
    """An ordinary object is not Fittable."""

    assert not mutable.is_fittable(_flat())
    assert not mutable.is_fittable(42)


def test_is_mutable_of_a_fittable_object() -> None:
    """A Fittable object is mutable."""

    assert mutable.is_mutable(_fittable())


def test_is_mutable_of_a_holder() -> None:
    """An object depending on a Fittable sub-object is mutable."""

    assert mutable.is_mutable(_holder())


def test_is_mutable_of_an_immutable_object() -> None:
    """An object with no Fittable sub-object is not mutable."""

    assert not mutable.is_mutable(_flat())
    assert not mutable.is_mutable(42)


def test_is_frozen_of_a_new_fittable() -> None:
    """A newly constructed Fittable has not been frozen."""

    assert not mutable.is_frozen(_fittable())


def test_is_frozen_of_an_immutable_object() -> None:
    """An immutable object counts as frozen, because it can never change."""

    assert mutable.is_frozen(_flat())
    assert mutable.is_frozen(42)


def test_freeze_returns_true_the_first_time() -> None:
    """freeze() reports whether it was this call that froze the object."""

    fov = _fittable()

    assert mutable.freeze(fov)
    assert not mutable.freeze(fov)


def test_freeze_of_an_immutable_object_returns_false() -> None:
    """An immutable object was already frozen, so nothing changes."""

    assert not mutable.freeze(_flat())


def test_freeze_reaches_the_sub_objects() -> None:
    """Freezing an object freezes all of its sub-objects too."""

    holder = _holder()
    mutable.freeze(holder)

    assert mutable.is_frozen(holder)
    assert mutable.is_frozen(holder.fov)


def test_mutable_names_of_a_holder() -> None:
    """The names of the mutable sub-objects are reported."""

    assert 'fov' in mutable.mutable_names(_holder())


def test_mutable_names_of_an_immutable_object() -> None:
    """An immutable object has no mutable sub-objects."""

    assert mutable.mutable_names(_flat()) == []


def test_unfrozen_names_before_and_after_freezing() -> None:
    """A sub-object drops out of the list once it is frozen."""

    holder = _holder()
    assert 'fov' in mutable.unfrozen_names(holder)

    mutable.freeze(holder)
    assert mutable.unfrozen_names(holder) == []


def test_version_starts_at_zero() -> None:
    """The version number starts at zero."""

    assert mutable.version(_fittable()) == 0


def test_version_increases_when_the_object_changes() -> None:
    """Setting the parameters of a Fittable increments its version."""

    fov = _fittable()
    before = mutable.version(fov)
    mutable.set_params(fov, (7., 8.))

    assert mutable.version(fov) > before


def test_version_of_an_immutable_object_is_zero() -> None:
    """An immutable object can never change, so its version stays at zero."""

    assert mutable.version(_flat()) == 0


def test_get_nparams_of_a_fittable() -> None:
    """A Fittable reports the number of parameters it carries."""

    assert mutable.get_nparams(_fittable()) == 2


def test_get_nparams_of_an_immutable_object() -> None:
    """An object with no parameters reports zero."""

    assert mutable.get_nparams(_flat()) == 0


def test_get_params_of_a_fittable() -> None:
    """The parameters are returned as a tuple."""

    assert mutable.get_params(_fittable()) == (1., 2.)


def test_get_params_of_an_immutable_object() -> None:
    """An object with no parameters returns an empty tuple."""

    assert mutable.get_params(_flat()) == ()


def test_set_params_changes_the_object() -> None:
    """set_params reports that the object changed, and the new values take effect."""

    fov = _fittable()

    assert mutable.set_params(fov, (7., 8.))
    assert mutable.get_params(fov) == (7., 8.)


def test_set_params_rejects_the_wrong_count() -> None:
    """A parameter list of the wrong length raises ValueError."""

    with pytest.raises(ValueError):
        mutable.set_params(_fittable(), (7.,))


def test_set_params_rejects_a_frozen_object() -> None:
    """A frozen object can no longer be modified."""

    fov = _fittable()
    mutable.freeze(fov)

    with pytest.raises(ValueError):
        mutable.set_params(fov, (7., 8.))


def test_set_params_reaches_a_sub_object() -> None:
    """Once a parameter order is set, a holder's parameters are its sub-objects'."""

    holder = _holder()
    mutable.set_param_order(holder, ['fov'])
    mutable.set_params(holder, (7., 8.))

    assert mutable.get_params(holder.fov) == (7., 8.)


def test_a_holder_has_no_parameters_until_an_order_is_set() -> None:
    """The parameters of the sub-objects are not exposed until they are ordered."""

    holder = _holder()

    assert mutable.get_nparams(holder) == 0
    assert mutable.get_params(holder) == ()


def test_needs_refresh_after_a_change() -> None:
    """A holder needs refreshing once one of its sub-objects has been modified."""

    holder = _holder()
    mutable.refresh(holder)
    assert not mutable.needs_refresh(holder)

    mutable.set_params(holder.fov, (7., 8.))
    assert mutable.needs_refresh(holder)


def test_refresh_clears_the_need_to_refresh() -> None:
    """Refreshing brings the object back up to date."""

    holder = _holder()
    mutable.set_params(holder.fov, (7., 8.))
    mutable.refresh(holder)

    assert not mutable.needs_refresh(holder)


def test_refresh_reports_whether_it_changed_anything() -> None:
    """refresh() returns True only when it actually updated the object."""

    holder = _holder()
    mutable.set_params(holder.fov, (7., 8.))

    assert mutable.refresh(holder)
    assert not mutable.refresh(holder)


def test_needs_refresh_of_a_refreshed_immutable_object() -> None:
    """An immutable object stays up to date once it has been refreshed."""

    fov = _flat()
    mutable.refresh(fov)

    assert not mutable.needs_refresh(fov)


def test_needs_refresh_of_a_plain_value() -> None:
    """A value that cannot hold cached information never needs refreshing."""

    assert not mutable.needs_refresh(42)


def test_get_param_order_defaults_to_empty() -> None:
    """An object with no parameters has an empty parameter order."""

    assert mutable.get_param_order(_flat()) == []


def test_set_param_order_records_the_order() -> None:
    """The named sub-objects define the order in which parameters are applied."""

    holder = _holder()
    mutable.set_param_order(holder, ['fov'])

    assert mutable.get_param_order(holder) == ['fov']


def test_set_param_order_rejects_an_unknown_name() -> None:
    """A name that is not a mutable sub-object raises ValueError."""

    with pytest.raises(ValueError):
        mutable.set_param_order(_holder(), ['not_a_subobject'])


def test_set_param_order_rejects_an_immutable_object() -> None:
    """An object that is not mutable has no parameter order to set."""

    with pytest.raises(ValueError, match='no attribute fov'):
        mutable.set_param_order(_flat(), ['fov'])


##########################################################################################
# The Mutable mix-in exposes the same API as methods
##########################################################################################

def test_mixin_is_frozen() -> None:
    """The method agrees with the function."""

    fov = _fittable()

    assert fov.is_frozen == mutable.is_frozen(fov)


def test_mixin_freeze() -> None:
    """Freezing through the method leaves the object frozen."""

    fov = _fittable()
    fov.freeze()

    assert fov.is_frozen


def test_mixin_params_and_nparams() -> None:
    """The methods agree with the functions."""

    fov = _fittable()

    assert fov.params == mutable.get_params(fov)
    assert fov.nparams == mutable.get_nparams(fov)


def test_mixin_set_params() -> None:
    """Setting parameters through the method changes the object."""

    fov = _fittable()
    fov.set_params((7., 8.))

    assert fov.params == (7., 8.)


def test_mixin_refresh() -> None:
    """Refreshing through the method clears the need to refresh."""

    holder = _holder()
    holder.fov.set_params((7., 8.))
    holder.refresh()

    assert not mutable.needs_refresh(holder)


def test_mixin_version() -> None:
    """The method agrees with the function."""

    fov = _fittable()

    assert fov.version == mutable.version(fov)

def test_set_param_order_can_only_be_called_once() -> None:
    """The order is part of the object's identity, so it is not redefined."""

    obj = _holder()
    mutable.set_param_order(obj, ['fov'])

    with pytest.raises(ValueError, match='parameter order was already defined'):
        mutable.set_param_order(obj, ['fov'])


def test_set_param_order_rejects_a_blank_name_on_an_unfittable_object() -> None:
    """A blank name marks where the object's own parameters go, so it must have some."""

    with pytest.raises(ValueError, match='object is not Fittable'):
        mutable.set_param_order(_holder(), [''])


def test_set_param_order_rejects_an_empty_list_of_names() -> None:
    """An order that names nothing leaves no parameters to fit."""

    with pytest.raises(ValueError, match='no fittable parameters'):
        mutable.set_param_order(_holder(), [])


def test_a_parameter_order_can_include_the_object_itself() -> None:
    """A blank name places the object's own parameters among its sub-objects'."""

    obj = OffsetFOV(_fittable(), uv_offset=(3., 4.))
    mutable.set_param_order(obj, ['fov', ''])

    assert mutable.get_nparams(obj) == 4
    assert mutable.get_params(obj) == (1., 2., 3., 4.)

    changed = mutable.set_params(obj, (5., 6., 7., 8.))

    assert changed
    assert mutable.get_params(obj) == (5., 6., 7., 8.)
    assert obj.fov.uv_offset.vals.tolist() == [5., 6.]


def test_the_versions_of_a_holder_name_its_mutable_sub_objects() -> None:
    """The version dictionary is keyed by the name of each mutable sub-object."""

    obj = _holder()

    assert sorted(mutable._versions(obj)) == sorted(mutable.mutable_names(obj))


def test_the_version_of_an_object_that_cannot_record_one_is_zero() -> None:
    """An object with no attribute dictionary has no version to keep."""

    assert mutable.version(()) == 0
    assert mutable._increment(()) == 0


def test_freezing_reaches_an_object_more_than_one_level_down() -> None:
    """A Fittable two levels below the object being frozen is frozen too."""

    inner = _fittable()
    obj = SliceFOV(SliceFOV(inner, (0, 0), (10, 10)), (0, 0), (5, 5))

    assert mutable.freeze(obj)
    assert mutable.is_frozen(inner)
    assert not mutable.freeze(obj)

##########################################################################################
