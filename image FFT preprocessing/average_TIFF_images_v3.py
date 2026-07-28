from pathlib import Path
import numpy as np
import tifffile as tiff
import tempfile
import shutil

# --- Configure matplotlib backend for non-interactive saving ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Define the Averaging Function ---
def process_tiff_stack_average(file_path):
    """Loads a 3D TIFF stack, averages it, and saves both 2D TIFF and PNG with a colorbar."""
    # Read the 3D stack
    stack = tiff.imread(file_path, out='memmap')
    
    # Calculate the average across axis 0
    averaged_image = np.mean(stack, axis=0).astype(np.float32)
    
    # Modify file_path to append "_col_scale_average" to the name
    base_name = f"{file_path.stem}_col_scale_average"
    tiff_path = file_path.with_name(f"{base_name}{file_path.suffix}")
    png_path = file_path.with_name(f"{base_name}.png")
    
    # 1. Save the 2D TIFF image
    tiff.imwrite(tiff_path, averaged_image)
    
    # --- 2. SAVE PNG WITH COLOUR SCALE ---
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    
    vmin, vmax = averaged_image.min(), averaged_image.max()
    if vmin == vmax:
        vmax += 1

    im = ax.imshow(averaged_image, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
    
    # Add color scale legend (colorbar)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    
    ax.axis('off')  # Hide tick marks/border
    fig.tight_layout()

    # Safe write via temporary local transit to prevent network issues
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_name = tmp.name

    try:
        fig.savefig(temp_name, bbox_inches='tight')
        plt.close(fig)  # Release RAM
        
        png_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(temp_name, str(png_path))
    except Exception as e:
        plt.close(fig)
        if Path(temp_name).exists():
            Path(temp_name).unlink()
        raise e


# --- Setup Paths (Using raw strings 'r') ---
input_base_path = Path(r"I:\Departamentos\Óptica\paulabp\master\TFM\MFBD images\cropped_shifted_originals")

telescopes = ["CS", "NOT"]
star_type = ["binarias", "dudosas-binarias", "simples"]

# --- Datasets ---
images_binarias_CS = ["CHR181_cropped_shifted_128.tif", "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_512.tif"]
images_dud_binarias_CS = ["hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_256.tif", "LAB4_cropped_shifted_128.tif", "STF1967_cropped_shifted_128.tif", "YSC8AB_cropped_shifted_128.tif"]
images_simples_CS = ["COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_128.tif", "FK384_cropped_shifted_128.tif"]
images_binarias_NOT = ["55Uma_NOT_cropped_shifted_128.tif", "CHR181_NOT_cropped_shifted_128.tif", "wds16289_NOT_cropped_shifted_512.tif"]
images_dud_binarias_NOT = ["wds14514_NOT_cropped_shifted_128.tif"]
images_simples_NOT = ["COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128.tif"]

# --- Loop Execution ---
for telescope in telescopes:
    input_path_folder = input_base_path / telescope

    for star in star_type:
        input_path_folder2 = input_path_folder / star
    
        if telescope == "CS" and star == "binarias":
            for image in images_binarias_CS:
                path_image = input_path_folder2 / image
                process_tiff_stack_average(path_image)

        if telescope == "CS" and star == "dudosas-binarias":
            for image in images_dud_binarias_CS:
                path_image = input_path_folder2 / image
                process_tiff_stack_average(path_image)

        if telescope == "CS" and star == "simples":
            for image in images_simples_CS:
                path_image = input_path_folder2 / image
                process_tiff_stack_average(path_image)

        if telescope == "NOT" and star == "binarias":
            for image in images_binarias_NOT:
                path_image = input_path_folder2 / image
                process_tiff_stack_average(path_image)

        if telescope == "NOT" and star == "dudosas-binarias":
            for image in images_dud_binarias_NOT:
                path_image = input_path_folder2 / image
                process_tiff_stack_average(path_image)

        if telescope == "NOT" and star == "simples":
            for image in images_simples_NOT:
                path_image = input_path_folder2 / image
                process_tiff_stack_average(path_image)