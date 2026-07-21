import numpy as np
import tifffile as tiff
from pathlib import Path
from collections import defaultdict
import tempfile
import shutil

# --- FREEZE PREVENTION PATCH ---
# We configure matplotlib to use a non-interactive backend.
# This prevents plt.show() or plt.imsave from opening GUI windows that block the loop.
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def find_complex_pairs_flexible(image_list, folder_path):
    """
    Intelligently scans the physical disk to find pairs, correcting variations
    in the suffix ordering of '_average' and '_apodized', while ignoring
    dimension variants to prevent filename mismatches.
    """
    if not folder_path.exists():
        return []
        
    # Read all actual .tif file names physically existing in the folder
    existing_files = {f.name for f in folder_path.glob("*.tif")}
    
    # 1. Identify the unique base names of the standard images (non-average files)
    standard_bases = set()
    for filename in image_list:
        if "average" not in filename:
            # Strip component suffixes to get the clean root base
            base = filename.replace("_FFT_real.tif", "").replace("_FFT_imaginary.tif", "")
            standard_bases.add(base)
            
    pairs = []
    for base in standard_bases:
        std_real = f"{base}_FFT_real.tif"
        std_imag = f"{base}_FFT_imaginary.tif"
        
        # 2. Generate all possible naming variations for the corresponding average file.
        # This handles cases where '_average' was saved before or after the '_apodized' flag.
        possible_roots = [
            base.replace("_apodized", "") + "_average_apodized",
            base + "_average",
            base.replace("_apodized", "_average_apodized")
        ]
        
        avg_real, avg_imag = None, None
        for root in possible_roots:
            test_real = f"{root}_FFT_real.tif"
            test_imag = f"{root}_FFT_imaginary.tif"
            if test_real in existing_files and test_imag in existing_files:
                avg_real = test_real
                avg_imag = test_imag
                break
                
        # If all 4 required real and imaginary parts exist physically on the disk, pair them
        if std_real in existing_files and std_imag in existing_files and avg_real and avg_imag:
            pairs.append({
                "base_name": base,
                "std_real": std_real,
                "std_imag": std_imag,
                "avg_real": avg_real,
                "avg_imag": avg_imag
            })
            
    return pairs

def safe_tiff_read(file_path):
    """
    Copies a TIFF file from a potentially slow network drive (I:) to the local C: drive,
    reads it ultra-fast into memory, and immediately cleans up the local copy.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Si ya está en local o pesa muy poco, lo leemos normal
    # Pero si está en red, creamos un puente temporal en local C:
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        temp_name = tmp.name
        
    try:
        # Copia rápida por bloques del archivo entero de la red a local C:
        shutil.copy2(str(file_path), temp_name)
        # Lectura ultra rápida desde el almacenamiento local
        data = tiff.imread(temp_name)
    finally:
        # Nos aseguramos al 100% de liberar el espacio en C: pase lo que pase
        if Path(temp_name).exists():
            Path(temp_name).unlink()
            
    return data

def safe_tiff_write(target_path, data):
    """
    Writes the TIFF file to the local temp directory first, 
    then transfers it to the network path to prevent partial write errors (OSError).
    """
    # Create a temporary file on the local machine (usually C:)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        temp_name = tmp.name
    
    try:
        # Fast local write (immune to network micro-interruptions)
        tiff.imwrite(temp_name, data.astype(np.float32))
        
        # Ensure target network directory structure exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic file copy transfer to the network drive
        shutil.move(temp_name, str(target_path))
    except Exception as e:
        if Path(temp_name).exists():
            Path(temp_name).unlink()
        raise e

def safe_png_write(target_path, array_data, cmap_name='viridis'):
    """Normalizes and safely saves a PNG using a temporary local transit to network."""
    if cmap_name in ['bwr', 'twilight']:
        vmax = max(abs(array_data.min()), abs(array_data.max()))
        vmin = -vmax if vmax > 0 else -1
        vmax = vmax if vmax > 0 else 1
    else:
        vmin, vmax = array_data.min(), array_data.max()
        if vmin == vmax:
            vmax += 1

    norm_data = (array_data - vmin) / (vmax - vmin)
    cmap = plt.get_cmap(cmap_name)
    rgba_image = (cmap(norm_data) * 255).astype(np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_name = tmp.name
        
    try:
        plt.imsave(temp_name, rgba_image)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(temp_name, str(target_path))
    except Exception as e:
        if Path(temp_name).exists():
            Path(temp_name).unlink()
        raise e

def calculations_complex_ops(pairs, input_folder, output_folder):
    """
    Reconstructs complex arrays, performs vectorized matrix math operations,
    and writes outputs cleanly. Uses safe local reads and writes to bypass network bottlenecks.
    """
    epsilon = 1e-08
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for pair in pairs:
        print(f"Processing Matrix Operations for: {pair['base_name']}")
        
        try:
            # Ahora leemos pasando por el "puente" local rápido
            std_r = safe_tiff_read(input_folder / pair['std_real']).astype(np.float32)
            std_i = safe_tiff_read(input_folder / pair['std_imag']).astype(np.float32)
            avg_r = safe_tiff_read(input_folder / pair['avg_real']).astype(np.float32)
            avg_i = safe_tiff_read(input_folder / pair['avg_imag']).astype(np.float32)
        except Exception as e:
            print(f"  -> Error reading components for {pair['base_name']}: {e}")
            continue

        complex_std = std_r + 1j * std_i
        complex_avg = avg_r + 1j * avg_i

        # --- 1. MULTIPLICACIÓN COMPLEJA ---
        complex_mult = complex_std * complex_avg
        
        # --- 2. DIVISIÓN COMPLEJA CON REGULARIZACIÓN ---
        avg_magnitude = np.abs(complex_avg)
        complex_avg_safe = np.copy(complex_avg)
        complex_avg_safe[avg_magnitude < epsilon] = epsilon
        
        complex_div = complex_std / complex_avg_safe

        # Comprobación de dimensiones (3D vs 2D) para la imagen PNG
        is_3d = complex_std.ndim == 3
        slice_idx = 0 if is_3d else ...
        base_stem = pair['base_name']

        # --- 3. COMPONENTES DE MULTIPLICACIÓN ---
        mult_components = {
            "real": np.real(complex_mult),
            "imaginary": np.imag(complex_mult),
            "magnitude": np.log(np.abs(complex_mult) + 1),
            "phase": np.angle(complex_mult)
        }
        for name, data in mult_components.items():
            tif_target = output_folder / f"{base_stem}_FFT_mult_{name}.tif"
            png_target = output_folder / f"{base_stem}_FFT_mult_{name}_slice0.png"
            
            # Se guarda el TIFF entero (mantiene las 3 dimensiones)
            safe_tiff_write(tif_target, data)
            
            # El PNG extrae solo la primera rebanada (slice 0) si es 3D
            cmap = 'bwr' if name in ['real', 'imaginary'] else ('twilight' if name == 'phase' else 'viridis')
            safe_png_write(png_target, data[slice_idx], cmap_name=cmap)

        # --- 4. COMPONENTES DE DIVISIÓN ---
        div_components = {
            "real": np.real(complex_div),
            "imaginary": np.imag(complex_div),
            "magnitude": np.log(np.abs(complex_div) + 1),
            "phase": np.angle(complex_div)
        }
        for name, data in div_components.items():
            tif_target = output_folder / f"{base_stem}_FFT_div_{name}.tif"
            png_target = output_folder / f"{base_stem}_FFT_div_{name}_slice0.png"
            
            # Se guarda el TIFF entero (mantiene las 3 dimensiones)
            safe_tiff_write(tif_target, data)
            
            # El PNG extrae solo la primera rebanada (slice 0) si es 3D
            cmap = 'bwr' if name in ['real', 'imaginary'] else ('twilight' if name == 'phase' else 'viridis')
            safe_png_write(png_target, data[slice_idx], cmap_name=cmap)

# --- Paths and Setup ---
input_base_path = Path(r"I:\Departamentos\Óptica\paulabp\master\TFM\MFBD images\FFT")
output_base_path = input_base_path / "new methods"

image_mapping = {
    ("CS", "binarias"): [
        "CHR181_cropped_shifted_128_apodized_FFT_imaginary.tif", 
        "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_512_apodized_FFT_imaginary.tif", 
        "CHR181_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_512_average_apodized_FFT_imaginary.tif", 
        "CHR181_cropped_shifted_128_apodized_FFT_real.tif", 
        "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_512_apodized_FFT_real.tif", 
        "CHR181_cropped_shifted_128_average_apodized_FFT_real.tif", 
        "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_512_average_apodized_FFT_real.tif" 
    ],
    ("CS", "dudosas-binarias"): [
        "LAB4_cropped_shifted_128_apodized_FFT_imaginary.tif", "STF1967_cropped_shifted_128_apodized_FFT_imaginary.tif", "YSC8AB_cropped_shifted_128_apodized_FFT_imaginary.tif", 
        "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_256_apodized_FFT_imaginary.tif", 
        "LAB4_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "STF1967_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "YSC8AB_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_256_average_apodized_FFT_imaginary.tif", 
        "LAB4_cropped_shifted_128_apodized_FFT_real.tif", "STF1967_cropped_shifted_128_apodized_FFT_real.tif", "YSC8AB_cropped_shifted_128_apodized_FFT_real.tif", 
        "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_256_apodized_FFT_real.tif", 
        "LAB4_cropped_shifted_128_average_apodized_FFT_real.tif", 
        "STF1967_cropped_shifted_128_average_apodized_FFT_real.tif", 
        "YSC8AB_cropped_shifted_128_average_apodized_FFT_real.tif", 
        "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_256_average_apodized_FFT_real.tif"
    ],
    ("CS", "simples"): [
        "COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_128_apodized_FFT_imaginary.tif", 
        "FK384_cropped_shifted_128_apodized_FFT_imaginary.tif", 
        "COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "FK384_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_128_apodized_FFT_real.tif", 
        "FK384_cropped_shifted_128_apodized_FFT_real.tif", 
        "COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_128_average_apodized_FFT_real.tif", 
        "FK384_cropped_shifted_128_average_apodized_FFT_real.tif" 
    ],
    ("NOT", "binarias"): [
        "55Uma_NOT_cropped_shifted_128_apodized_FFT_imaginary.tif", "CHR181_NOT_cropped_shifted_128_apodized_FFT_imaginary.tif", "wds16289_NOT_cropped_shifted_512_apodized_FFT_imaginary.tif", 
        "55Uma_NOT_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "CHR181_NOT_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "wds16289_NOT_cropped_shifted_512_average_apodized_FFT_imaginary.tif", 
        "55Uma_NOT_cropped_shifted_128_apodized_FFT_real.tif", "CHR181_NOT_cropped_shifted_128_apodized_FFT_real.tif", "wds16289_NOT_cropped_shifted_512_apodized_FFT_real.tif", 
        "55Uma_NOT_cropped_shifted_128_average_apodized_FFT_real.tif", 
        "CHR181_NOT_cropped_shifted_128_average_apodized_FFT_real.tif", 
        "wds16289_NOT_cropped_shifted_512_average_apodized_FFT_real.tif"
    ],
    ("NOT", "dudosas-binarias"): [
        "wds14514_NOT_cropped_shifted_128_apodized_FFT_imaginary.tif", 
        "wds14514_NOT_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "wds14514_NOT_cropped_shifted_128_apodized_FFT_real.tif", 
        "wds14514_NOT_cropped_shifted_128_average_apodized_FFT_real.tif"
    ],
    ("NOT", "simples"): [
        "COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128_apodized_FFT_imaginary.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128_apodized_FFT_imaginary.tif", 
        "COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128_average_apodized_FFT_imaginary.tif", 
        "COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128_apodized_FFT_real.tif", "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128_apodized_FFT_real.tif", 
        "COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128_average_apodized_FFT_real.tif", 
        "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_128_average_apodized_FFT_real.tif"
    ]
}

# --- Loop Execution ---
for (telescope, star), images in image_mapping.items():
    input_folder = input_base_path / telescope / star
    output_folder = output_base_path / telescope / star
    
    # Run a dynamic file scan directly on the network share (I:\)
    pairs = find_complex_pairs_flexible(images, input_folder)
    if pairs:
        calculations_complex_ops(pairs, input_folder, output_folder)