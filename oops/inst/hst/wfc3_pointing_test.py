import numpy as np
import numpy.ma as ma
import scipy.ndimage.filters as filters
import scipy.fftpack as fftpack
import pylab
import oops
import oops.inst.hst as hst

# A quick-and-dirty image correlation function
def correlate2d(image, model, normalize=False):
    """Correlate the image with the model; normalization to [-1.1] is optional.
    """
    image_fft = fftpack.fft2(image)
    model_fft = fftpack.fft2(model)
    corr = np.real(fftpack.ifft2(image_fft * np.conj(model_fft)))
    if normalize:
        corr /= np.sqrt(np.sum(image**2) * np.sum(model**2))
    return corr

# Create the snapshot object
filespec = "test_data/hst/ibht02v5q_flt.fits"
snapshot = hst.from_file(filespec)
pylab.imshow(snapshot.data)

# Give the image a quick cleanup
blur = filters.median_filter(snapshot.data, 9)
flat = (snapshot.data - blur)
image = flat.clip(-1000,6000).astype("float")
pylab.imshow(image)

# Define the model image as two ansas of the epsilon ring
meshgrid = oops.Meshgrid.for_fov(snapshot.fov, swap=True)
bp = oops.Backplane(snapshot, meshgrid)

epsilon = bp.border_atop(("ring_radius", "epsilon_ring"),
                         51149.32).vals.astype("float")
pylab.imshow(np.maximum(epsilon * image.max(), image))

# Subtract out the saturated columns
disk = bp.where_intercepted("uranus").vals
pylab.imshow(disk)
disk[:] = np.any(disk, axis=0)
epsilon[disk] = 0.
pylab.imshow(epsilon)

# Blur and then sharpen the model so it looks more like the image
model = filters.gaussian_filter(epsilon,1)
blur = filters.median_filter(model, 9)
model -= blur
pylab.imshow(model)

# Locate the pixel offsets with the highest correlation between image and model
corr = correlate2d(image, model)
# pylab.imshow(corr)

(vmax,umax) = np.where(corr == corr.max())
umax = umax[0]
vmax = vmax[0]

if umax > image.shape[0]/2: umax -= image.shape[0]

if vmax > image.shape[1]/2: vmax -= image.shape[1]

print (umax,vmax)

# Update the FOV object with the new pointing offset
snapshot.fov = oops.fov.Offset(snapshot.fov, (umax,vmax))

# Define the model image as two ansas of the epsilon ring
meshgrid = oops.Meshgrid.for_fov(snapshot.fov, swap=True)
bp = oops.Backplane(snapshot, meshgrid)

epsilon = bp.border_atop(("ring_radius", "epsilon_ring"),
                         51149.32).vals.astype("float")
pylab.imshow(np.maximum(epsilon * image.max(), image))
