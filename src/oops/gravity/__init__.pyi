##########################################################################################
# oops/gravity/__init__.pyi
##########################################################################################
"""Type stub for :mod:`oops.gravity`.

The `src` tree carries no inline annotations, so type information for public symbols is
published here instead. Only package stubs exist, so a name is annotated when it is
imported from the package that exports it and not when it is imported from the module
that defines it. The stub describes the shape of the API exactly: every public name, its
parameters, which of them are keyword-only, and which have defaults. Types are given
where they are unambiguous and are `Any` elsewhere.
"""

from oops.gravity.gravity_ import Gravity as Gravity
from oops.gravity.oblategravity import OblateGravity as OblateGravity

__all__ = ['Gravity', 'OblateGravity']

##########################################################################################
