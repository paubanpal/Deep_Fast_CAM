# pip install ipython (then type ipython on the terminal)
#import cv2  # pip install opencv-python
import tifffile as tiff # pip install tifffile
from pathlib import Path
import numpy as np
import torchmfbd
import torch
import matplotlib.pyplot as pl

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

    #n_images = [10, 20]
    n_images = [10, 20, 50, 90, 200, 500]

    for i in n_images:
        # frames shape: [i, H, W] -> 3D tensor
        frames = torch.tensor(img_stack[0:i, ...], dtype = torch.float32)
        frames /= frames.max()

        # Deconvolution process
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / 'config_CS_yaml.yaml'
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
        orig_shape = frames[:, :, :].shape  # [10, 128, 128]
        
        # TRY THIS: Pull the reconstructed scene object directly
        # If it's a method, use decSI.get_object(). If it's a property, use decSI.object_scene
        if hasattr(decSI, 'get_object'):
            deconv_output = decSI.get_object()
        elif hasattr(decSI, 'object'):
            deconv_output = decSI.object
        else:
            # Fallback if it is inside obj but structured as [modes, frames]
            # We take the mean or look for the diffraction-limited version
            deconv_output = decSI.obj_diffraction[0] if hasattr(decSI, 'obj_diffraction') else decSI.obj[0]

        # Ensure it is detached and converted cleanly
        if isinstance(deconv_output, torch.Tensor):
            deconv_output = deconv_output.detach().cpu().numpy()
            
        # Debug print to your HPC terminal so you can verify the shape is now 300x300
        print(f"--- SUCCESS! Reconstructed Image Shape for {i} frames: {deconv_output.shape} ---")

        # Check if the output is a raw 2D image (128, 128) and expand for uniform plotting loops
        if deconv_output.ndim == 2:
            deconv_output = deconv_output.unsqueeze(0)
            
        # Check if the output is a raw 2D image (128, 128) and expand for uniform plotting loops
        if hasattr(deconv_output, 'ndim') and deconv_output.ndim == 2:
            # We handle it safely depending on whether it's PyTorch or NumPy
            if isinstance(deconv_output, torch.Tensor):
                deconv_output = deconv_output.unsqueeze(0)
            else:
                deconv_output = np.expand_dims(deconv_output, axis=0)
            
        # SAFE CONVERSION: Only call .cpu() if it's actually a PyTorch Tensor
        if isinstance(deconv_output, torch.Tensor):
            obj.append(deconv_output.cpu().numpy())
        else:
            obj.append(deconv_output) # It's already a NumPy array!
        # ========================================================

        # 3. Unpatchify your original background frames for the visualization comparison
        reconstructed_back = PyTorchPatchify.unpatchify(
            frames_patches, 
            output_shape=orig_shape, 
            patch_size=128, 
            stride_size=50, 
            apodization=0
        )
        frames_back.append(reconstructed_back.cpu().numpy())

        # 4. Set up the plotting window boundaries based on our real 128x128 shape
        npix = orig_shape[1] 
        fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
        
        # Plot 2 original raw frames vs our reconstructed deconvolution matrix
        for j in range(2):
            # Row 0: Original degraded short-exposure frames
            ax[0, j].imshow(frames[j, :, :].cpu().numpy(), cmap='gray')
            ax[0, j].set_title(f"Degraded Frame {j+1}")
            
            # Row 1: Deconvolved output (mapped safely to your 128x128 array grid)
            if obj[0].shape[0] > j:
                ax[1, j].imshow(obj[0][j, 0:npix, 0:npix], cmap='gray')
            else:
                ax[1, j].imshow(obj[0][0, 0:npix, 0:npix], cmap='gray')
            ax[1, j].set_title(f"Deconvolved Scene Object")
        
        # 5. Save the default internal library weights (the 44x10 matrix)
        name = path_image.stem + '_' + str(i) + '_MFBD_weights' + path_image.suffix
        final_path = path_folder / name
        decSI.write(final_path)

        # ========================================================
        # NEW ADDITION: Save the actual 128x128 restored image!
        # ========================================================
        # obj[0] contains the numpy array of shape (1, 128, 128)
        # We slice obj[0][0] to get the clean 2D image matrix (128x128)
        final_image_data = obj[0][0]
        
        # Build a new name for the actual image file
        image_name = path_image.stem + '_' + str(i) + '_MFBD_RECONSTRUCTED' + path_image.suffix
        image_final_path = path_folder / image_name
        
        # Save using the tiff package you imported at the top of your script
        tiff.imwrite(image_final_path, final_image_data.astype(np.float32))


# We start from a set of frames of shape (n_sequences, n_objects, n_frames, n_pixel, n_pixel)

# Pathlib handles the slashes and spacing logic for you (for spaces and accents in path)
#path = Path("I:/Departamentos/Óptica/paulabp/master/TFM/Lucky Imaging Miguel/imagenes LI/simples/FK384_cropped.tif")    # path en windows
hpc_base_path = Path("/scratch/paulabp/TFM/images/CS/original/")
#star_type = ["binarias"]
star_type = ["binarias", "dudosas-binarias", "simples"]
images_binarias = ["CHR181_cropped.tif"]
images_dud_binarias = ["LAB4_cropped.tif", "STF1967_cropped.tif", "YSC8AB_cropped.tif"]
images_simples = ["COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped.tif", "FK384_cropped.tif"]

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