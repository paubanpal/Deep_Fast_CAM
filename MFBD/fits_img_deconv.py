# pip install ipython (then type ipython on the terminal)
#import cv2  # pip install opencv-python
from astropy.io import fits # pip install astropy
from pathlib import Path
import numpy as np
import torchmfbd


# We start from a set of frames of shape (n_sequences, n_objects, n_frames, n_pixel, n_pixel)

# Pathlib handles the slashes and spacing logic for you (for spaces and accents in path)
path = Path("I:/Departamentos/Óptica/paulabp/master/TFM/Lucky Imaging Miguel/imagenes LI/simples/KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped.fits")
#print(path.exists())

# Load the image
# Opening a FITS file
with fits.open(path) as hdul:
    # FITS files are organized into "HDU" (Header Data Units)
    #data = hdul[0].data  # This gives you the NumPy array
    #header = hdul[0].header

    img_stack = np.array([hdu.data for hdu in hdul[1:] if hdu.data is not None])

print(img_stack.shape) # This should now show (N, 128, 128)
#print("Image size: ", data.shape)
#print("Header: ", header)

# Images saved as .FITS by ImageJ once the original image has been cropped only save one of the images, so they have finally been all saved as .tif
# DO NOT USE THIS FILE
with fits.open(path) as hdul:
    hdul.info()
    for i, hdu in enumerate(hdul):
        data_shape = hdu.data.shape if hdu.data is not None else "No Data"
        print(f"HDU {i}: Type={type(hdu).__name__}, Shape={data_shape}")

n_seq = 1      # Number of sequences
n_obj = 1      # Number of objects
n_frm = img_stack.shape[0]     # Number of frames per object/sequence
h, w = img_stack.shape[1], img_stack.shape[2]

# Reshape to MFBD format
img_stack_MFBD = img_stack.reshape((n_seq, n_obj, n_frm, h, w))
print(img_stack_MFBD.shape)

# Deconvolution process
deconv = torchmfbd.Deconvolution('config_CS_yaml.yaml')

deconv.deconvolve(infer_object=False,   # If False, the object is inferred using the analytic solution given by the Wiener filter. Otherwise, the object is inferred by the optimizer.
                 optimizer='adam',  # "adam" (first order) or "lbfgs" (second order L-BFGS, that is more memory and time consuming but more efficient in terms of number of iterations)
                 simultaneous_sequences=16, # The number of patches to deconvolve simultaneously. If you have plenty of VRAM, you can increase this number to speed up the deconvolution.
                 n_iterations=20)

name = path.stem + '_MFBD' + path.suffix
deconv.write(name)


"""
Once the deconvolution is finished, several attributes are available in the deconvolution object:

    obj: The deconvolved objects.

    obj_diffraction: The deconvolved objects convolved with the diffraction-limited PSF.

    psf: The inferred PSFs

    degraded: The object convolved with the inferred PSFs. They can be used to check the quality of the deconvolution because they should be similar to the input frames.

A final call to the write method will save the deconvolved objects and the modes to a FITS file.
"""