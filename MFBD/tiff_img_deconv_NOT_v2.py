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
        Slices a PyTorch tensor into overlapping patches and flattens them.
        Expects tensor shape: (batch/sequence, channels, height, width) 
        or adjusted for your specific 5D 'frames' tensor dimensions.
        """
        # Assuming frames shape structure from your code allows extracting H and W from last 2 dims
        B, C, _, H, W = tensor.shape 
        
        # Reshape or squeeze to 4D for unfold if needed
        # We process frame by frame or batch them together
        tensor_4d = tensor.view(-1, C, H, W) 
        
        # Unfold extracts sliding local blocks
        patches = tensor_4d.unfold(2, patch_size, stride_size).unfold(3, patch_size, stride_size)
        # Permute and reshape to flatten the patch sequences matching patchify's structure
        # Shape: (Num_Patches, Channels, patch_size, patch_size)
        patches = patches.contiguous().view(-1, C, patch_size, patch_size)
        return patches

    @staticmethod
    def unpatchify(patches, output_shape, patch_size=64, stride_size=50, apodization=0):
        """
        Reconstructs the original tensor from overlapping patches using Fold.
        Applies basic linear/cosine windowing if apodization > 0 to smooth edges.
        """
        B, C, _, H, W = output_shape
        # Create a PyTorch Fold operation
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=(patch_size, patch_size), stride=(stride_size, stride_size))
        
        # Prepare patches back to the shape Fold expects: (BxC, patch_size*patch_size, Num_Patches)
        # Note: Depending on torchmfbd's output, you may need to adjust view parameters here
        num_patches = ( (H - patch_size) // stride_size + 1 ) * ( (W - patch_size) // stride_size + 1 )
        
        # If apodization is used, we generate a basic weight mask to handle overlapping division
        # If apodization=0, we just divide by a counting matrix to average the overlaps
        patches_reshaped = patches.view(B * C, num_patches, patch_size * patch_size).permute(0, 2, 1)
        
        reconstructed = fold(patches_reshaped)
        
        # Create normalization mask to account for overlap accumulation
        ones = torch.ones_like(patches_reshaped)
        col_mask = fold(ones)
        
        # Divide by overlap counts to smooth out intensity spikes
        final_tensor = reconstructed / (col_mask + 1e-8)
        return final_tensor.view(B, C, H, W)


def read_and_deconvolve(path_image, path_folder):
    # Load the image
    img_stack = tiff.imread(path_image)
    #print(img_stack.shape)

    n_images = [10, 20]
    #n_images = [10, 20, 50, 90, 200, 500]

    for i in n_images:
        frames = torch.tensor(img_stack[0:i, ...], dtype = torch.float32)
        frames /= frames.max()

        # Deconvolution process
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / 'config_NOT_yaml.yaml'
        decSI = torchmfbd.Deconvolution(str(config_path))

        
        # patches = extract_patches_2d(frames, (64, 64))
        # decSI.add_frames(patches[j], id_object = 0, id_diversity = 0, diversity = 0.0)

        frames_patches = PyTorchPatchify.patchify(frames[:, :, :, :, :], patch_size=64, stride_size=50)
        decSI.add_frames(frames_patches, id_object=0, id_diversity=0, diversity=0.0)


        # Patchify and add the frames
        #decSI.add_frames(frames[None, ...], id_object = 0, id_diversity = 0, diversity = 0.0)


        decSI.deconvolve(infer_object=False,   # If False, the object is inferred using the analytic solution given by the Wiener filter. Otherwise, the object is inferred by the optimizer.
                 optimizer='adam',  # "adam" (first order) or "lbfgs" (second order L-BFGS, that is more memory and time consuming but more efficient in terms of number of iterations)
                 simultaneous_sequences=150, # The number of patches to deconvolve simultaneously. If you have plenty of VRAM, you can increase this number to speed up the deconvolution.
                 n_iterations=250)
        
        obj = []
        frames_back = []
        for j in range(1):
            orig_shape = frames[:, j, :, :, :].shape
    
            obj.append(PyTorchPatchify.unpatchify(decSI.obj[i], output_shape=orig_shape, patch_size=64, stride_size=50, apodization=6).cpu().numpy())
            frames_back.append(PyTorchPatchify.unpatchify(frames_patches[i], output_shape=orig_shape, patch_size=64, stride_size=50, apodization=0).cpu().numpy())

        npix = obj[0][0, :, :].shape[0]
        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for j in range(2):
            ax[0, j].imshow(frames[0, j, 0, 0:npix, 0:npix])
            ax[1, j].imshow(obj[i][0, :, :])
        
        # fig, ax = pl.subplots(nrows = 1, ncols = 5, figsize = (15, 5))
        # for j in range(2):
        #     ax[j].imshow(frames[j, ...], cmap = 'gray')

        # ax[-1].imshow(decSI.obj[0][0, ...].cpu().numpy(), cmap = 'gray')

        name = path_image.stem + '_' + str(i) + '_MFBD' + path_image.suffix
        final_path = path_folder / name
        decSI.write(final_path)


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