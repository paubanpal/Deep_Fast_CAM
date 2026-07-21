from pathlib import Path
import numpy as np
import tifffile as tiff

# --- Define the Averaging Function ---
def process_tiff_stack_average(file_path):
    """Loads a 3D TIFF stack, averages it, and saves it in the same folder with '_average' appended."""
    # Read the 3D stack
    stack = tiff.imread(file_path, out='memmap')
    
    # Calculate the average across axis 0
    averaged_image = np.mean(stack, axis=0)
    
    # Modify file_path to append "_average" to the name
    # e.g., converts 'image.tif' -> 'image_average.tif'
    new_filename = f"{file_path.stem}_average{file_path.suffix}"
    file_path = file_path.with_name(new_filename)
    
    # Save the 2D image using the updated file_path
    tiff.imwrite(file_path, averaged_image.astype(np.float32))
    #print(f"Saved averaged image to: {file_path.name}")


# --- Setup Paths (Using raw strings 'r') ---
input_base_path = Path(r"I:\Departamentos\Óptica\paulabp\master\TFM\MFBD images\cropped_shifted_originals")

telescopes = ["CS", "NOT"]
star_type = ["binarias", "dudosas-binarias", "simples"]

# --- Datasets ---
images_binarias_CS = ["CHR181_cropped_shifted.tif", "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted.tif"]
images_dud_binarias_CS = ["LAB4_cropped_shifted.tif", "STF1967_cropped_shifted.tif", "YSC8AB_cropped_shifted.tif", "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted.tif"]
images_simples_CS = ["COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted.tif", "FK384_cropped_shifted.tif"]
images_binarias_NOT = ["55Uma_NOT_cropped_shifted.tif", "CHR181_NOT_cropped_shifted.tif", "wds16289_NOT_cropped_shifted.tif"]
images_dud_binarias_NOT = ["wds14514_NOT_cropped_shifted.tif"]
images_simples_NOT = ["COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted.tif"]

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