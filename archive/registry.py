################################################################################
# oops/registry.py: Global Frame and Path Registry
################################################################################

################################################################################
# Frame Registry
################################################################################

# NOTE: This file has the property that it does NOT need to import the Frame
# class. This is necessary so that modules that must load earlier (such as
# Event and Transform) will not have circular dependencies.

J2000 = None
FRAME_REGISTRY = {}
FRAME_CACHE = {}
FRAME_CLASS = None
TEMPORARY_FRAME_ID = 10000

#===============================================================================
# frame_lookup
#===============================================================================
def frame_lookup(key):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a frame from the registry given one of its keys.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return FRAME_REGISTRY[key]
#===============================================================================



#===============================================================================
# as_frame
#===============================================================================
def as_frame(frame):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a Frame object given the registered name or the object itself.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if frame is None: return None

    if isinstance(frame, FRAME_CLASS): return frame

    return FRAME_REGISTRY[frame]
#===============================================================================



#===============================================================================
# as_primary_frame
#===============================================================================
def as_primary_frame(frame):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a the primary definition of a Frame object.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if frame is None: return None

    if not isinstance(frame, FRAME_CLASS):
        frame = FRAME_REGISTRY[frame]

    return FRAME_CACHE[frame.wayframe]
#===============================================================================



#===============================================================================
# as_frame_id
#===============================================================================
def as_frame_id(frame):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a Frame ID given the object or a registered ID.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if frame is None: return None

    if isinstance(frame, FRAME_CLASS): return frame.frame_id

    return frame
#===============================================================================



#===============================================================================
# temporary_frame_id
#===============================================================================
def temporary_frame_id():
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a temporary frame ID.
    
    This is assigned once and never re-used.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global TEMPORARY_FRAME_ID

    while True:
        TEMPORARY_FRAME_ID += 1
        frame_id = "TEMPORARY_" + str(TEMPORARY_FRAME_ID)
        if not FRAME_REGISTRY.has_key(frame_id):
            return frame_id
#===============================================================================



#===============================================================================
# initialize_frame_registry
#===============================================================================
def initialize_frame_registry():
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Initialize the registry.
    
    It is not generally necessary to call this function, but it can be used
    to reset the registry for purposes of debugging.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global J2000, FRAME_REGISTRY, FRAME_CACHE

    J2000 = None
    FRAME_REGISTRY.clear()
    FRAME_CACHE.clear()
    FRAME_CLASS.initialize_registry()
#===============================================================================



#===============================================================================
# connect_frames
#===============================================================================
def connect_frames(target, reference):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a Frame object that transforms between two frames.

    Input:
        target      a Frame object or the registered ID of the destination
                    frame.
        reference   a Frame object or the registered ID of the starting frame.

    The shape of the connected frame will will be the result of broadcasting
    the shapes of the target and reference.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    target = as_frame(target)
    return target.connect_to(reference)
#===============================================================================



################################################################################
# Path Registry
################################################################################

# NOTE: This file has the property that it does NOT need to import the Path
# class. This is necessary so that modules that must load earlier (such as
# Event and Transform) will not have circular dependencies.

SSB = None
PATH_REGISTRY = {}
PATH_CACHE = {}
PATH_CLASS = None
TEMPORARY_PATH_ID = 10000

#===============================================================================
# path_lookup
#===============================================================================
def path_lookup(key):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a path from the registry given one of its keys.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return PATH_REGISTRY[key]
#===============================================================================



#===============================================================================
# as_path
#===============================================================================
def as_path(path):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a Path object given the registered name or the object itself.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if path is None: return None

    if isinstance(path, PATH_CLASS): return path

    return PATH_REGISTRY[path]
#===============================================================================



#===============================================================================
# as_primary_path
#===============================================================================
def as_primary_path(path):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a the primary definition of a Path object.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if path is None: return None

    if not isinstance(path, PATH_CLASS):
        path = PATH_REGISTRY[path]

    return PATH_CACHE[path.waypoint]
#===============================================================================



#===============================================================================
# as_path_id
#===============================================================================
def as_path_id(path):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a path ID given the object or a registered ID.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if path is None: return None

    if isinstance(path, PATH_CLASS): return path.path_id

    return path
#===============================================================================



#===============================================================================
# temporary_path_id
#===============================================================================
def temporary_path_id():
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a temporary path ID. This is assigned once and never re-used.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global TEMPORARY_PATH_ID

    while True:
        TEMPORARY_PATH_ID += 1
        path_id = "TEMPORARY_" + str(TEMPORARY_PATH_ID)
        if path_id not in PATH_REGISTRY:
            return path_id
#===============================================================================



#===============================================================================
# initialize_path_registry
#===============================================================================
def initialize_path_registry():
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Initialize the registry.
    
    It is not generally necessary to call this function, but it can be used
    to reset the registry for purposes of debugging.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global SSB, PATH_REGISTRY, PATH_CACHE

    SSB = None
    PATH_REGISTRY.clear()
    PATH_CACHE.clear()
    PATH_CLASS.initialize_registry()
#===============================================================================



#===============================================================================
# connect_paths
#===============================================================================
def connect_paths(target, origin, frame=None):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a path that connects two paths.
    
    The returned path creates event objects in which vectors point from any
    origin path to any target path, using any coordinate frame.

    Input:
        target      the Path object or ID of the target path.
        origin      the Path object or ID of the origin path.
        frame       the Frame object of ID of the coordinate frame to use;
                    use None for the default frame of the origin.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    target = as_path(target)
    return target.connect_to(origin, frame)
#===============================================================================



################################################################################
# Body Registry
################################################################################
# A dictionary associating the names of solar system bodies with their Body
# objects.

BODY_REGISTRY = {}
BODY_CLASS = None

#===============================================================================
# is_body
#===============================================================================
def is_body(item):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return True if the item is a Body object.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return isinstance(item, BODY_CLASS)
#===============================================================================



#===============================================================================
# body_lookup
#===============================================================================
def body_lookup(key):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a body from the registry given its name.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global BODY_REGSITRY
    return BODY_REGISTRY[key]
#===============================================================================



#===============================================================================
# body_exists
#===============================================================================
def body_exists(key):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return True if the body's name exists in the registry.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global BODY_REGSITRY
    return BODY_REGISTRY.has_key(key)
#===============================================================================



#===============================================================================
# as_body
#===============================================================================
def as_body(body):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a body object given the registered name or the object itself.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if is_body(body): return body
    return body_lookup(body)
#===============================================================================



#===============================================================================
# as_body_name
#===============================================================================
def as_body_name(body):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a body name given the registered name or the object itself.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if is_body(body): return body.name
    return body
#===============================================================================



#===============================================================================
# initialize_body_registry
#===============================================================================
def initialize_body_registry():
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Initialize the registry.
    
    It is not generally necessary to call this function, but it can be used
    to reset the registry for purposes of debugging.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global BODY_REGISTRY
    BODY_REGISTRY = {}
#===============================================================================



################################################################################
# General Functions
################################################################################

#===============================================================================
# initialize
#===============================================================================
def initialize():
    initialize_path_registry()
    initialize_frame_registry()
    initialize_body_registry()
#===============================================================================



################################################################################

# There are no unit tests for registry because it is amply tested in all other
# modules.
