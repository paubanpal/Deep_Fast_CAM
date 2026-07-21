import numpy as np
import tifffile as tiff
from pathlib import Path
from collections import defaultdict

def find_image_pairs(image_list):
    """
    Groups images into pairs based on their filenames, matching the standard 
    version with its corresponding '_average' version.
    """
    pairs_dict = defaultdict(list)
    for filename in image_list:
        base_key = filename.replace("_average", "")
        pairs_dict[base_key].append(filename)
    return [tuple(pair) for pair in pairs_dict.values() if len(pair) == 2]

def calculations_stack_fft(pairs, input_base_path):
    base_dir = Path(input_base_path)
    epsilon = 1e-05  
    
    for standard_name, average_name in pairs:
        full_standard_path = base_dir / standard_name
        full_average_path = base_dir / average_name
        
        print(f"Processing: {standard_name}")
        
        # Load the TIFF images as memory maps
        stack_standard = tiff.imread(full_standard_path, out='memmap')
        img_average = tiff.imread(full_average_path, out='memmap')
        
        # Convert average to float32 and inject epsilon
        img_average_f32 = img_average.astype(np.float32)
        img_average_safe = img_average_f32 + epsilon
    
        # Determine the output filenames
        mult_output_path = base_dir / standard_name.replace(".tif", "_multiply.tif")
        div_output_path = base_dir / standard_name.replace(".tif", "_division.tif")
        
        if stack_standard.ndim == 3:
            num_frames, height, width = stack_standard.shape
            
            # Open both writers simultaneously for a single-pass processing loop
            with tiff.TiffWriter(str(mult_output_path)) as mult_writer, \
                 tiff.TiffWriter(str(div_output_path)) as div_writer:
                     
                for i in range(num_frames):
                    frame_f32 = stack_standard[i].astype(np.float32)
                    
                    # 1. Multiplication
                    frame_mult = frame_f32 * img_average_f32
                    mult_writer.write(frame_mult, contiguous=True)
                    
                    # 2. Division
                    frame_div = frame_f32 / img_average_safe
                    div_writer.write(frame_div, contiguous=True)
                    
            print(f"  -> Saved process stacks to: {base_dir.name}")
            
        else:
            # Fallback for 2D images
            standard_f32 = stack_standard.astype(np.float32)
            
            # 1. Multiplication (Fixed: wrapped paths in str() for network compatibility)
            result_mult = standard_f32 * img_average_f32
            tiff.imwrite(str(mult_output_path), result_mult)
            
            # 2. Division
            result_div = standard_f32 / img_average_safe
            tiff.imwrite(str(div_output_path), result_div)
            
            #print(f"  -> Saved 2D processed images to: {base_dir.name}")

# --- Paths and Setup ---
input_base_path = Path(r"I:\Departamentos\Óptica\paulabp\master\TFM\MFBD images\FFT")
telescopes = ["CS", "NOT"]
star_type = ["binarias", "dudosas-binarias", "simples"]

images_binarias_CS = ["CHR181_cropped_shifted_FFT.tif", "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_FFT.tif", "CHR181_cropped_shifted_average_FFT.tif", "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_average_FFT.tif"]
images_dud_binarias_CS = ["LAB4_cropped_shifted_FFT.tif", "STF1967_cropped_shifted_FFT.tif", "YSC8AB_cropped_shifted_FFT.tif", "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_FFT.tif", "LAB4_cropped_shifted_average_FFT.tif", "STF1967_cropped_shifted_average_FFT.tif", "YSC8AB_cropped_shifted_average_FFT.tif", "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_average_FFT.tif"]
images_simples_CS = ["COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_FFT.tif", "FK384_cropped_shifted_FFT.tif", "COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_average_FFT.tif", "FK384_cropped_shifted_average_FFT.tif"]
images_binarias_NOT = ["55Uma_NOT_cropped_shifted_FFT.tif", "CHR181_NOT_cropped_shifted_FFT.tif", "wds16289_NOT_cropped_shifted_FFT.tif", "55Uma_NOT_cropped_shifted_average_FFT.tif", "CHR181_NOT_cropped_shifted_average_FFT.tif", "wds16289_NOT_cropped_shifted_average_FFT.tif"]
images_dud_binarias_NOT = ["wds14514_NOT_cropped_shifted_FFT.tif", "wds14514_NOT_cropped_shifted_average_FFT.tif"]
images_simples_NOT = ["COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_FFT.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_FFT.tif", "COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_average_FFT.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_average_FFT.tif"]

# --- Loop Execution ---
for telescope in telescopes:
    input_path_folder = input_base_path / telescope

    for star in star_type:
        input_path_folder2 = input_path_folder / star

        if telescope == "CS" and star == "binarias":
            pairs = find_image_pairs(images_binarias_CS)
            calculations_stack_fft(pairs, input_path_folder2)

        elif telescope == "CS" and star == "dudosas-binarias":
            pairs = find_image_pairs(images_dud_binarias_CS)
            calculations_stack_fft(pairs, input_path_folder2)

        elif telescope == "CS" and star == "simples":
            pairs = find_image_pairs(images_simples_CS)
            calculations_stack_fft(pairs, input_path_folder2)

        elif telescope == "NOT" and star == "binarias":
            pairs = find_image_pairs(images_binarias_NOT)
            calculations_stack_fft(pairs, input_path_folder2)

        elif telescope == "NOT" and star == "dudosas-binarias":
            pairs = find_image_pairs(images_dud_binarias_NOT)
            calculations_stack_fft(pairs, input_path_folder2)

        elif telescope == "NOT" and star == "simples":
            pairs = find_image_pairs(images_simples_NOT)
            calculations_stack_fft(pairs, input_path_folder2)