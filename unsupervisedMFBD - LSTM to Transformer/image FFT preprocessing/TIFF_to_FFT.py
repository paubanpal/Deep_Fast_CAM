import numpy as np
import tifffile as tiff
from pathlib import Path

def process_tiff_stack_fft(input_path, output_path):
    # 1. Load the TIFF stack (reads as a 3D NumPy array: [slices, height, width])
    # Added out='memmap' to prevent network drive freezing on large files
    stack = tiff.imread(input_path, out='memmap')
    print(f"Loaded stack shape: {stack.shape} (Dimensions: {stack.ndim})")
    
    # CASE A: If it's a 3D stack, loop through slices as originally planned
    if stack.ndim == 3:
        # Initialize an empty array to store the magnitude spectrum
        # Using float32 or float64 to handle the continuous values of the FFT
        fft_stack = np.zeros(stack.shape, dtype=np.float32)
        
        # Loop through each slice and compute the 2D FFT
        for i in range(stack.shape[0]):
            slice_data = stack[i]
            
            # Compute the 2D Fast Fourier Transform
            f_transform = np.fft.fft2(slice_data)
            
            # Shift the zero-frequency component to the center of the spectrum
            f_shift = np.fft.fftshift(f_transform)
            
            # Calculate the magnitude spectrum (absolute value)
            magnitude_spectrum = np.abs(f_shift)
            
            # Optional: Apply log scaling so you can actually see the patterns 
            # (FFT magnitudes span several orders of magnitude)
            # We add 1 to avoid log(0)
            log_spectrum = np.log(magnitude_spectrum + 1)
            
            fft_stack[i] = log_spectrum
            
    # CASE B: If it's already a single 2D image, process it directly without a loop
    elif stack.ndim == 2:
        #print(f"--> Notice: {input_path.name} is a single 2D image, processing directly.")
        # Compute the 2D Fast Fourier Transform
        f_transform = np.fft.fft2(stack)
        
        # Shift the zero-frequency component to the center of the spectrum
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate the magnitude spectrum (absolute value)
        magnitude_spectrum = np.abs(f_shift)
        
        # Optional: Apply log scaling so you can actually see the patterns 
        # We add 1 to avoid log(0)
        fft_stack = np.log(magnitude_spectrum + 1)
        
    else:
        print(f"Skipping {input_path.name}: Unexpected array dimension ({stack.ndim}).")
        return

    # Ensure the parent directory exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the processed stack as a new TIFF file
    # We normalize or cast back to a standard format if needed, but tifffile 
    # handles float32 natively, which preserves the FFT data accurately.
    tiff.imwrite(output_path, fft_stack.astype(np.float32))
    #print(f"Saved FFT stack to: {output_path.name}\n")


# Pathlib handles the slashes and spacing logic for you (for spaces and accents in path)
input_base_path = Path(r"I:\Departamentos\Óptica\paulabp\master\TFM\MFBD images\cropped_shifted_originals")
output_base_path = Path(r"I:\Departamentos\Óptica\paulabp\master\TFM\MFBD images\FFT")
telescopes = ["CS", "NOT"]
#star_type = ["binarias"]
star_type = ["binarias", "dudosas-binarias", "simples"]
images_binarias_CS = ["CHR181_cropped_shifted.tif", "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted.tif", "CHR181_cropped_shifted_average.tif", "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_average.tif"]
images_dud_binarias_CS = ["LAB4_cropped_shifted.tif", "STF1967_cropped_shifted.tif", "YSC8AB_cropped_shifted.tif", "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted.tif", "LAB4_cropped_shifted_average.tif", "STF1967_cropped_shifted_average.tif", "YSC8AB_cropped_shifted_average.tif", "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_average.tif"]
images_simples_CS = ["COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted.tif", "FK384_cropped_shifted.tif", "COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_average.tif", "FK384_cropped_shifted_average.tif"]
images_binarias_NOT = ["55Uma_NOT_cropped_shifted.tif", "CHR181_NOT_cropped_shifted.tif", "wds16289_NOT_cropped_shifted.tif", "55Uma_NOT_cropped_shifted_average.tif", "CHR181_NOT_cropped_shifted_average.tif", "wds16289_NOT_cropped_shifted_average.tif"]
images_dud_binarias_NOT = ["wds14514_NOT_cropped_shifted.tif", "wds14514_NOT_cropped_shifted_average.tif"]
images_simples_NOT = ["COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted.tif", "COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_average.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_average.tif"]

# --- Loop Execution ---
for telescope in telescopes:
    input_path_folder = input_base_path / telescope
    output_path_folder = output_base_path / telescope

    for star in star_type:
        input_path_folder2 = input_path_folder / star
        output_path_folder2 = output_path_folder / star
    
        # Helper logic to dynamically generate the "_FFT" destination path
        # It takes an image path, adds _FFT, and targets the output folder
        def get_fft_destination(image_name, target_folder):
            # Create a temporary Path object for the filename string
            p = Path(image_name)
            new_name = f"{p.stem}_FFT{p.suffix}" # e.g., "image_FFT.tif"
            return target_folder / new_name

        if telescope == "CS" and star == "binarias":
            for image in images_binarias_CS:
                path_image = input_path_folder2 / image
                destination = get_fft_destination(image, output_path_folder2)
                process_tiff_stack_fft(path_image, destination)

        if telescope == "CS" and star == "dudosas-binarias":
            for image in images_dud_binarias_CS:
                path_image = input_path_folder2 / image
                destination = get_fft_destination(image, output_path_folder2)
                process_tiff_stack_fft(path_image, destination)

        if telescope == "CS" and star == "simples":
            for image in images_simples_CS:
                path_image = input_path_folder2 / image
                destination = get_fft_destination(image, output_path_folder2)
                process_tiff_stack_fft(path_image, destination)

        if telescope == "NOT" and star == "binarias":
            for image in images_binarias_NOT:
                path_image = input_path_folder2 / image
                destination = get_fft_destination(image, output_path_folder2)
                process_tiff_stack_fft(path_image, destination)

        if telescope == "NOT" and star == "dudosas-binarias":
            for image in images_dud_binarias_NOT:
                path_image = input_path_folder2 / image
                destination = get_fft_destination(image, output_path_folder2)
                process_tiff_stack_fft(path_image, destination)

        if telescope == "NOT" and star == "simples":
            for image in images_simples_NOT:
                path_image = input_path_folder2 / image
                destination = get_fft_destination(image, output_path_folder2)
                process_tiff_stack_fft(path_image, destination)