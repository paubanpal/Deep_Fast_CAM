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

    n_images = 10
    #n_images = [10, 20, 50, 90, 200, 500]

    for i in n_images:
        frames = torch.tensor(img_stack[0:i, ...], dtype = torch.float32)
        frames /= frames.max()

        # Deconvolution process
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / 'config_NOT_yaml.yaml'
        decSI = torchmfbd.Deconvolution(str(config_path))

        # Patchify and add the frames
        decSI.add_frames(frames[None, ...], id_object = 0, id_diversity = 0, diversity = 0.0)


        decSI.deconvolve(infer_object=False,   # If False, the object is inferred using the analytic solution given by the Wiener filter. Otherwise, the object is inferred by the optimizer.
                 optimizer='adam',  # "adam" (first order) or "lbfgs" (second order L-BFGS, that is more memory and time consuming but more efficient in terms of number of iterations)
                 simultaneous_sequences=16, # The number of patches to deconvolve simultaneously. If you have plenty of VRAM, you can increase this number to speed up the deconvolution.
                 n_iterations=20)
        
        fig, ax = pl.subplots(nrows = 1, ncols = 5, figsize = (15, 5))
        for i in range(4):
            ax[i].imshow(frames[i, ...], cmap = 'gray')

        ax[-1].imshow(decSI.obj[0][0, ...].cpu().numpy(), cmap = 'gray')

        name = path_image.stem + '_' + i + '_MFBD' + path_image.suffix
        final_path = path_folder / name
        decSI.write(final_path)


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