from cspice import *
import cspice

def bods2c(name):
    result = cspice.bods2c(name)
    if result[1]: return result[0]
    raise LookupError('SPICE body not recognized: "' + name + '"')

def bodn2c(name):
    result = cspice.bodn2c(name)
    if result[1]: return result[0]
    raise LookupError("SPICE body name not recognized: " + name)

def bodc2n(id):
    result = cspice.bodc2n(id)
    if result[1]: return result[0]
    raise LookupError("SPICE body ID not recognized: " + id)

def bodfnd(id,item):
    return (cspice.bodfnd(id,item) != 0)

def bodvcd(id,item):
    if cspice.bodfnd(id,item.upper()):
	return cspice.bodvcd(id,item.upper())

    raise LookupError("SPICE item not found for body ID " + str(id) +
                      ": " + item)
def bodvrd(name,item):
    id = bodn2c(name)
    if cspice.bodfnd(id,item.upper()):
	return cspice.bodvrd(name,item.upper())

    raise LookupError("SPICE item not found for body \"" + name +
                      "\": " + item)


