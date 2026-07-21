# pip install ipython (then type ipython on the terminal)
#import cv2  # pip install opencv-python
import tifffile as tiff # pip install tifffile
from pathlib import Path
import numpy as np
import torchmfbd
import torch
import matplotlib.pyplot as pl



def read_and_deconvolve(path_image, path_folder):
    # Load the image
    img_stack = tiff.imread(path_image)
    #print(img_stack.shape)

    n_seq = 1      # Number of sequences
    n_obj = 1      # Number of objects
    n_frm = img_stack.shape[0]     # Number of frames per object/sequence
    h, w = img_stack.shape[1], img_stack.shape[2]

    n_images = [10, 20, 50, 90, 200, 500]

    for i in n_images:
        # Reshape to MFBD format
        img_stack_MFBD = img_stack.reshape((n_seq, n_obj, n_frm, h, w))
        img_stack_MFBD = img_stack_MFBD[:, :, :i, :, :]
        #print(img_stack_MFBD.shape)

        # Deconvolution process
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / 'config_NOT_yaml.yaml'
        deconv = torchmfbd.Deconvolution(str(config_path))

        # Convert your NumPy array to a PyTorch Tensor
        torch_tensor_stack = torch.from_numpy(img_stack_MFBD).float()

        # Squeeze out the 'n_obj' dimension (dimension index 1) to make it 4D:
        # (1, 1, 10, 128, 128) becomes (1, 10, 128, 128)
        torch_tensor_stack_4d = torch_tensor_stack.squeeze(1)
        deconv.add_frames(torch_tensor_stack_4d)

        deconv.deconvolve(infer_object=False,   # If False, the object is inferred using the analytic solution given by the Wiener filter. Otherwise, the object is inferred by the optimizer.
                 optimizer='adam',  # "adam" (first order) or "lbfgs" (second order L-BFGS, that is more memory and time consuming but more efficient in terms of number of iterations)
                 simultaneous_sequences=16, # The number of patches to deconvolve simultaneously. If you have plenty of VRAM, you can increase this number to speed up the deconvolution.
                 n_iterations=20)

        name = path_image.stem + '_' + i + '_MFBD' + path_image.suffix
        final_path = path_folder / name
        deconv.write(final_path)


# We start from a set of frames of shape (n_sequences, n_objects, n_frames, n_pixel, n_pixel)

# Pathlib handles the slashes and spacing logic for you (for spaces and accents in path)
#path = Path("I:/Departamentos/Óptica/paulabp/master/TFM/Lucky Imaging Miguel/imagenes LI/simples/FK384_cropped.tif")    # path en windows
hpc_base_path = Path("/scratch/paulabp/TFM/images/NOT/original/")
star_type = ["binarias", "dudosas-binarias", "simples"]
images_binarias = ["55Uma_NOT_cropped.tif", "CHR181_NOT_cropped.tif"]
images_dud_binarias = ["wds14514_NOT_cropped.tif"]
images_simples = ["COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped.tif"]

for star in star_type:
    path_folder = hpc_base_path / star
    
    if star == "binarias":
        for image in images_binarias:
            path_image = path_folder / image
            #path = Path("I:\Departamentos\Óptica\paulabp\master\TFM\Lucky Imaging Miguel\imagenes LI\simples\FK384_cropped.tif")
            #print(path.exists())

            read_and_deconvolve(path_image, path_folder)
    if star == "dudosas-binarias":
        for image in images_dud_binarias:
            path_image = path_folder / image
            #path = Path("I:\Departamentos\Óptica\paulabp\master\TFM\Lucky Imaging Miguel\imagenes LI\simples\FK384_cropped.tif")
            #print(path.exists())

            read_and_deconvolve(path_image, path_folder)
    if star == "simples":
        for image in images_simples:
            path_image = path_folder / image
            #path = Path("I:\Departamentos\Óptica\paulabp\master\TFM\Lucky Imaging Miguel\imagenes LI\simples\FK384_cropped.tif")
            #print(path.exists())

            read_and_deconvolve(path_image, path_folder)

    


"""
Once the deconvolution is finished, several attributes are available in the deconvolution object:

    obj: The deconvolved objects.

    obj_diffraction: The deconvolved objects convolved with the diffraction-limited PSF.

    psf: The inferred PSFs

    degraded: The object convolved with the inferred PSFs. They can be used to check the quality of the deconvolution because they should be similar to the input frames.

A final call to the write method will save the deconvolved objects and the modes to a FITS file.
"""