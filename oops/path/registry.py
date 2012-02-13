################################################################################
# oops/path/registry.py: Global Path Registry
################################################################################

# The global registry is keyed in two ways. If the key is a string, then
# this constitutes the primary definition of the path. The origin_id
# of the path must already be in the registry, and the new path's ID
# cannot be in the registry.

# We also create secondary definitions of a path, where it is defined
# relative to a different reference frame and/or with respect to a different
# origin. These are entered into the registry keyed twice, by a tuple:
#   (path_id, origin_id)
# and a triple:
#   (path_id, origin_id, frame_id).
# This saves the effort of re-creating paths used repeatedly.

# NOTE: This file has the property that it does NOT need to import the Path
# class. This is necessary so that modules that must load earlier (such as
# Event and Transform) will not have circular dependencies.

SSB = None
REGISTRY = {}
PATH_CLASS = None

def lookup(key): return REGISTRY[key]

def as_path(path):
    """Returns a Path object given the registered name or the object
    itself."""

    if path is None: return None

    try:
        test = path.path_id
        return path
    except AttributeError:
        return REGISTRY[path]

def as_id(path):
    """Returns a path ID given the object or a registered ID."""

    if path is None: return None

    try:
        return path.path_id
    except AttributeError:
        return path

def as_primary(path):
    """Returns the primary definition of a Path object, based on a
    registered name or a Path object."""

    try:
        return REGISTRY[path.path_id]
    except AttributeError:
        return REGISTRY[path]

def is_id(item):
    """Returns True if the item is a valid path ID."""

    abbr = item.__class__.__name__[0:3]
    return abbr in ("int", "str")

def is_path(item):
    """Returns true if the item is a Path object."""

    global PATH_CLASS
    return isinstance(item, PATH_CLASS)

def initialize_registry():
    """Initializes the registry. It is not generally necessary to call this
    function, but it can be used to reset the registry for purposes of
    debugging."""

    PATH_CLASS.initialize_registry()

def unregister_frame(frame_id):
    """Removes any explicit reference to this frame from the path registry.
    It does not affect any paths that might use this frame internally."""

    PATH_CLASS.unregister_frame(frame_id)

################################################################################
# Path Generator
################################################################################

def connect(target, origin, frame="J2000"):
    """Returns a path that creates event objects in which vectors point
    from any origin path to any target path, using any coordinate frame.

    Input:
        target      the Path object or ID of the target path.
        origin      the Path object or ID of the origin path.
        frame       the Frame object of ID of the coordinate frame to use;
                    use None for the default frame of the origin.
    """

    global PATH_CLASS
    return PATH_CLASS.connect(target, origin, frame)

################################################################################
