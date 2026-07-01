# pip install ipython (then type ipython on the terminal)
#import cv2  # pip install opencv-python
import tifffile as tiff # pip install tifffile
from pathlib import Path
import numpy as np
import torchmfbd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as pl
from sklearn.feature_extraction.image import extract_patches_2d
import scipy
import tqdm
import skimage
import sklearn
import nvitop
import yaml
import einops
import dict_hash

class PyTorchPatchify:
    @staticmethod
    def patchify(tensor, patch_size=64, stride_size=50):
        """
        Slices a 3D PyTorch tensor [C, H, W] into overlapping patches and flattens them.
        """
        # Expecting a 3D tensor: Channels/Batch, Height, Width
        C, H, W = tensor.shape 
        
        # Add a dummy batch dimension to make it 4D for the unfold operation [1, C, H, W]
        tensor_4d = tensor.unsqueeze(0) 
        
        # Unfold extracts sliding local blocks
        patches = tensor_4d.unfold(2, patch_size, stride_size).unfold(3, patch_size, stride_size)
        
        # Permute and reshape to flatten the patch sequences
        # Shape becomes: (Num_Patches, C, patch_size, patch_size)
        patches = patches.contiguous().view(-1, C, patch_size, patch_size)
        return patches

    @staticmethod
    def unpatchify(patches, output_shape, patch_size=64, stride_size=50, apodization=0):
        """
        Reconstructs the original 3D tensor [C, H, W] from overlapping patches using Fold.
        """
        C, H, W = output_shape
        
        # PyTorch Fold operation expects an explicit batch dimension, so we treat C as B*C
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=(patch_size, patch_size), stride=(stride_size, stride_size))
        
        num_patches = ((H - patch_size) // stride_size + 1) * ((W - patch_size) // stride_size + 1)
        
        # Prepare patches back to the shape Fold expects: (C, patch_size*patch_size, Num_Patches)
        patches_reshaped = patches.view(C, num_patches, patch_size * patch_size).permute(0, 2, 1)
        
        reconstructed = fold(patches_reshaped)
        
        # Create normalization mask to account for overlap accumulation
        ones = torch.ones_like(patches_reshaped)
        col_mask = fold(ones)
        
        # Divide by overlap counts to smooth out intensity spikes
        final_tensor = reconstructed / (col_mask + 1e-8)
        
        # Remove any extra squeeze dimensions to match original 3D shape [C, H, W]
        return final_tensor.view(C, H, W)


def read_and_deconvolve(path_image, path_folder):
    # Load the image
    img_stack = tiff.imread(path_image)

    n_images = [10, 20]
    #n_images = [10, 20, 50, 90, 200, 500]

    for i in n_images:
        # frames shape: [i, H, W] -> 3D tensor
        frames = torch.tensor(img_stack[0:i, ...], dtype = torch.float32)
        frames /= frames.max()

        # Deconvolution process
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / 'config_NOT_yaml.yaml'
        decSI = torchmfbd.Deconvolution(str(config_path))

        # Create patches using size 128
        frames_patches = PyTorchPatchify.patchify(frames[:, :, :], patch_size=128, stride_size=50)
        decSI.add_frames(frames_patches, id_object=0, id_diversity=0, diversity=0.0)

        decSI.deconvolve(infer_object=False,   
                         optimizer='adam',  
                         simultaneous_sequences=150, 
                         n_iterations=250)
        
        obj = []
        frames_back = []
        
        orig_shape = frames[:, :, :].shape  # Expecting [num_images, H, W]
        
        # FIX 1: Map decSI.obj[0] directly. It is already standard image shape data!
        # We just ensure it is shaped as [num_images, H, W] or whatever shape it provides natively.
        deconv_output = decSI.obj[0].detach()
        if deconv_output.ndim == 2:
            # If it returns a single 2D composite image, expand it to match the plotting expected format
            deconv_output = deconv_output.unsqueeze(0)
            
        obj.append(deconv_output.cpu().numpy())
        
        # FIX 2: Reconstruct the raw background frames_patches cleanly
        reconstructed_back = PyTorchPatchify.unpatchify(
            frames_patches, 
            output_shape=orig_shape, 
            patch_size=128, 
            stride_size=50, 
            apodization=0
        )
        frames_back.append(reconstructed_back.cpu().numpy())

        # Pull plot visualization boundaries
        npix = orig_shape[1] # Use height of the frame
        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
        
        # FIX 3: Clean, robust 3D matrix visualization plotting loops
        for j in range(2):
            ax[0, j].imshow(frames[j, :, :].cpu().numpy(), cmap='gray')
            
            # Checks if obj contains multiple frames or just 1 frame to prevent out-of-bounds indexing
            if obj[0].shape[0] > j:
                ax[1, j].imshow(obj[0][j, :, :], cmap='gray')
            else:
                ax[1, j].imshow(obj[0][0, :, :], cmap='gray') # Fallback to first deconv item
        
        # Save output
        name = path_image.stem + '_' + str(i) + '_MFBD' + path_image.suffix
        final_path = path_folder / name
        decSI.write(final_path)

# def read_and_deconvolve(path_image, path_folder):
#     # Load the image
#     img_stack = tiff.imread(path_image)
#     #print(img_stack.shape)

#     n_images = [10, 20]
#     #n_images = [10, 20, 50, 90, 200, 500]

#     for i in n_images:
#         frames = torch.tensor(img_stack[0:i, ...], dtype = torch.float32)
#         frames /= frames.max()

#         # Deconvolution process
#         script_dir = Path(__file__).resolve().parent
#         config_path = script_dir / 'config_NOT_yaml.yaml'
#         decSI = torchmfbd.Deconvolution(str(config_path))

        
#         # patches = extract_patches_2d(frames, (64, 64))
#         # decSI.add_frames(patches[j], id_object = 0, id_diversity = 0, diversity = 0.0)

#         frames_patches = PyTorchPatchify.patchify(frames[:, :, :], patch_size=128, stride_size=50)
#         decSI.add_frames(frames_patches, id_object=0, id_diversity=0, diversity=0.0)


#         # Patchify and add the frames
#         #decSI.add_frames(frames[None, ...], id_object = 0, id_diversity = 0, diversity = 0.0)


#         decSI.deconvolve(infer_object=False,   # If False, the object is inferred using the analytic solution given by the Wiener filter. Otherwise, the object is inferred by the optimizer.
#                  optimizer='adam',  # "adam" (first order) or "lbfgs" (second order L-BFGS, that is more memory and time consuming but more efficient in terms of number of iterations)
#                  simultaneous_sequences=150, # The number of patches to deconvolve simultaneously. If you have plenty of VRAM, you can increase this number to speed up the deconvolution.
#                  n_iterations=250)
        
#         obj = []
#         frames_back = []
#         for j in range(1):
#             orig_shape = frames[:, :, :].shape
    
#             obj.append(PyTorchPatchify.unpatchify(decSI.obj[i], output_shape=orig_shape, patch_size=128, stride_size=50, apodization=6).cpu().numpy())
#             frames_back.append(PyTorchPatchify.unpatchify(frames_patches[i], output_shape=orig_shape, patch_size=64, stride_size=50, apodization=0).cpu().numpy())

#         npix = obj[0][0, :, :].shape[0]
#         fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
#         for j in range(2):
#             ax[0, j].imshow(frames[0, j, 0, 0:npix, 0:npix])
#             ax[1, j].imshow(obj[i][0, :, :])
        
#         # fig, ax = pl.subplots(nrows = 1, ncols = 5, figsize = (15, 5))
#         # for j in range(2):
#         #     ax[j].imshow(frames[j, ...], cmap = 'gray')

#         # ax[-1].imshow(decSI.obj[0][0, ...].cpu().numpy(), cmap = 'gray')

#         name = path_image.stem + '_' + str(i) + '_MFBD' + path_image.suffix
#         final_path = path_folder / name
#         decSI.write(final_path)


# We start from a set of frames of shape (n_sequences, n_objects, n_frames, n_pixel, n_pixel)

# Pathlib handles the slashes and spacing logic for you (for spaces and accents in path)
#path = Path("I:/Departamentos/Óptica/paulabp/master/TFM/Lucky Imaging Miguel/imagenes LI/simples/FK384_cropped.tif")    # path en windows
hpc_base_path = Path("/scratch/paulabp/TFM/images/NOT/original/")
star_type = ["binarias"]
#star_type = ["binarias", "dudosas-binarias", "simples"]
images_binarias = ["55Uma_NOT_cropped.tif"]
#images_binarias = ["55Uma_NOT_cropped.tif", "CHR181_NOT_cropped.tif"]
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