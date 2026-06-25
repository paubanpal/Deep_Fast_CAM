# pip install ipython (then type ipython on the terminal)
#import cv2  # pip install opencv-python
import tifffile as tiff # pip install tifffile
from pathlib import Path
import numpy as np
import torchmfbd
import torch


# We start from a set of frames of shape (n_sequences, n_objects, n_frames, n_pixel, n_pixel)

# Pathlib handles the slashes and spacing logic for you (for spaces and accents in path)
#path = Path("I:/Departamentos/Óptica/paulabp/master/TFM/Lucky Imaging Miguel/imagenes LI/simples/FK384_cropped.tif")    # path en windows
hpc_base_path = Path("/scratch/paulabp/TFM/images/NOT/original/")
path_folder = hpc_base_path / "binarias"
path = path_folder / "55Uma_NOT_cropped.tif"
#path = Path("I:\Departamentos\Óptica\paulabp\master\TFM\Lucky Imaging Miguel\imagenes LI\simples\FK384_cropped.tif")
print(path.exists())

# Load the image
img_stack = tiff.imread(path)
print(img_stack.shape)

n_seq = 1      # Number of sequences
n_obj = 1      # Number of objects
n_frm = img_stack.shape[0]     # Number of frames per object/sequence
h, w = img_stack.shape[1], img_stack.shape[2]

# Reshape to MFBD format
img_stack_MFBD = img_stack.reshape((n_seq, n_obj, n_frm, h, w))
img_stack_MFBD = img_stack_MFBD[:, :, :10, :, :]
print(img_stack_MFBD.shape)

# Deconvolution process
script_dir = Path(__file__).resolve().parent
config_path = script_dir / 'config_NOT_yaml.yaml'
deconv = torchmfbd.Deconvolution(str(config_path))

# Convert your NumPy array to a PyTorch Tensor
torch_tensor_stack = torch.from_numpy(img_stack_MFBD).float()
# Hand over the PyTorch Tensor instead of the raw NumPy array
deconv.frames = torch_tensor_stack
# Tell the framework which object index each frame corresponds to.
# If all frames belong to 'object1', set them all to index 0:
deconv.ind_object = [0] * 10
# Map all 10 frames to diversity index 0
deconv.ind_diversity = [0] * 10
# Initialize the physical diversity tracking values
# We pass a PyTorch tensor with a single 0.0 value representing the focus state.
deconv.diversity = torch.tensor([0.0])

deconv.deconvolve(infer_object=False,   # If False, the object is inferred using the analytic solution given by the Wiener filter. Otherwise, the object is inferred by the optimizer.
                 optimizer='adam',  # "adam" (first order) or "lbfgs" (second order L-BFGS, that is more memory and time consuming but more efficient in terms of number of iterations)
                 simultaneous_sequences=16, # The number of patches to deconvolve simultaneously. If you have plenty of VRAM, you can increase this number to speed up the deconvolution.
                 n_iterations=20)

name = path.stem + 'MFBD' + path.suffix
deconv.write(name)


"""
Once the deconvolution is finished, several attributes are available in the deconvolution object:

    obj: The deconvolved objects.

    obj_diffraction: The deconvolved objects convolved with the diffraction-limited PSF.

    psf: The inferred PSFs

    degraded: The object convolved with the inferred PSFs. They can be used to check the quality of the deconvolution because they should be similar to the input frames.

A final call to the write method will save the deconvolved objects and the modes to a FITS file.
"""