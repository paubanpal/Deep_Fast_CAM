import numpy as np
import tifffile as tiff
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import window  # pip install scikit-image

def save_array_as_png(array_data, output_path, cmap_name='viridis'):
    """
    Helper function to normalize and save a 2D array as a beautiful 8-bit PNG image 
    using a Matplotlib colormap so it can be opened easily in standard photo viewers.
    """
    # Determine bounds for normalization
    if cmap_name in ['bwr', 'twilight']:
        # For symmetric parts (real, imag, or phase), center around zero or use absolute ranges
        vmax = max(abs(array_data.min()), abs(array_data.max()))
        vmin = -vmax if vmax > 0 else -1
        vmax = vmax if vmax > 0 else 1
    else:
        vmin, vmax = array_data.min(), array_data.max()
        if vmin == vmax:  # Prevent divide-by-zero for flat regions
            vmax += 1

    # Normalize data between 0 and 1
    norm_data = (array_data - vmin) / (vmax - vmin)
    
    # Apply colormap to get an RGBA image, then convert to 8-bit unsigned integer (0-255)
    cmap = plt.get_cmap(cmap_name)
    rgba_image = (cmap(norm_data) * 255).astype(np.uint8)
    
    # Save using matplotlib's image writer
    plt.imsave(output_path, rgba_image)

def process_tiff_stack_fft(input_path, output_folder, apodize=True):
    """
    Processes a TIFF image/stack, apodizes it, computes its FFT, and saves raw (.tif)
    and visualization (.png) versions for Magnitude, Real, Imaginary, and Phase components.
    For 3D stacks, only the first slice is exported to PNG.
    """
    try:
        stack = tiff.imread(str(input_path))
    except Exception as e:
        print(f"Error loading {input_path.name}: {e}")
        return

    print(f"Processing: {input_path.name} | Shape: {stack.shape}")
    is_3d = stack.ndim == 3
    
    # Standardize dimensions to a 3D loop even if it's a single 2D image
    if stack.ndim == 2:
        working_stack = stack[np.newaxis, ...]
    elif stack.ndim == 3:
        working_stack = stack
    else:
        print(f"Skipping {input_path.name}: Unexpected dimensions ({stack.ndim}).")
        return

    num_slices, height, width = working_stack.shape
    magnitude_stack = np.zeros_like(working_stack, dtype=np.float32)
    real_stack = np.zeros_like(working_stack, dtype=np.float32)
    imag_stack = np.zeros_like(working_stack, dtype=np.float32)
    phase_stack = np.zeros_like(working_stack, dtype=np.float32)

    # --- APODIZATION MATRIX SETUP ---
    if apodize:
        win_2d = window('tukey', (height, width))
    else:
        win_2d = np.ones((height, width))

    # Loop through each slice and compute the 2D FFT
    for i in range(num_slices):
        slice_data = working_stack[i] * win_2d
        
        # Compute the 2D FFT and center it
        f_transform = np.fft.fft2(slice_data)
        f_shift = np.fft.fftshift(f_transform)
        
        # Extract all 4 components
        real_stack[i] = np.real(f_shift)
        imag_stack[i] = np.imag(f_shift)
        magnitude_stack[i] = np.log(np.abs(f_shift) + 1)
        phase_stack[i] = np.angle(f_shift)  # True Phase Map in radians (-pi to +pi)

    # Squeeze back down if it was originally a simple 2D image
    if not is_3d:
        magnitude_stack = magnitude_stack[0]
        real_stack = real_stack[0]
        imag_stack = imag_stack[0]
        phase_stack = phase_stack[0]

    # Ensure output directories exist using pure pathlib syntax
    output_folder.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    suffix = "_apodized" if apodize else ""

    # --- 1. SAVE RAW DATA AS TIFF (Full Stack) ---
    mag_tif_path = output_folder / f"{stem}{suffix}_FFT_magnitude.tif"
    real_tif_path = output_folder / f"{stem}{suffix}_FFT_real.tif"
    imag_tif_path = output_folder / f"{stem}{suffix}_FFT_imaginary.tif"
    phase_tif_path = output_folder / f"{stem}{suffix}_FFT_phase.tif"

    tiff.imwrite(mag_tif_path, magnitude_stack.astype(np.float32))
    tiff.imwrite(real_tif_path, real_stack.astype(np.float32))
    tiff.imwrite(imag_tif_path, imag_stack.astype(np.float32))
    tiff.imwrite(phase_tif_path, phase_stack.astype(np.float32))

    # --- 2. SAVE VISUALIZED AS PNG ---
    if is_3d:
        # Save only the first slice (index 0) from the 3D data array
        save_array_as_png(magnitude_stack[0], output_folder / f"{stem}{suffix}_FFT_magnitude_slice0.png", cmap_name='viridis')
        save_array_as_png(real_stack[0], output_folder / f"{stem}{suffix}_FFT_real_slice0.png", cmap_name='bwr')
        save_array_as_png(imag_stack[0], output_folder / f"{stem}{suffix}_FFT_imaginary_slice0.png", cmap_name='bwr')
        save_array_as_png(phase_stack[0], output_folder / f"{stem}{suffix}_FFT_phase_slice0.png", cmap_name='twilight')
    else:
        # Standard 2D file save
        save_array_as_png(magnitude_stack, output_folder / f"{stem}{suffix}_FFT_magnitude.png", cmap_name='viridis')
        save_array_as_png(real_stack, output_folder / f"{stem}{suffix}_FFT_real.png", cmap_name='bwr')
        save_array_as_png(imag_stack, output_folder / f"{stem}{suffix}_FFT_imaginary.png", cmap_name='bwr')
        save_array_as_png(phase_stack, output_folder / f"{stem}{suffix}_FFT_phase.png", cmap_name='twilight')

    # --- 3. PLOT FOR ONSCREEN DISPLAY ---
    plot_fft_components(
        magnitude_stack[0] if is_3d else magnitude_stack,
        real_stack[0] if is_3d else real_stack,
        imag_stack[0] if is_3d else imag_stack,
        phase_stack[0] if is_3d else phase_stack,
        title=f"FFT Breakdown{suffix.replace('_', ' ')}: {stem}"
    )

def plot_fft_components(magnitude, real, imaginary, phase, title=""):
    """Generates a 2x2 grid plot of all four FFT components."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Top Left: Magnitude plot
    im0 = axes[0, 0].imshow(magnitude, cmap='viridis')
    axes[0, 0].set_title("Log Magnitude Spectrum")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Top Right: Phase Map (Ranges cleanly from -pi to +pi)
    im1 = axes[0, 1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title("Phase Map (radians)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Bottom Left: Real part
    vmax_real = max(abs(real.min()), abs(real.max()))
    im2 = axes[1, 0].imshow(real, cmap='bwr', vmin=-vmax_real, vmax=vmax_real)
    axes[1, 0].set_title("Real Part")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Bottom Right: Imaginary part
    vmax_imag = max(abs(imaginary.min()), abs(imaginary.max()))
    im3 = axes[1, 1].imshow(imaginary, cmap='bwr', vmin=-vmax_imag, vmax=vmax_imag)
    axes[1, 1].set_title("Imaginary Part")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    for row in axes:
        for ax in row:
            ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# --- Setup Paths & Collections ---
input_base_path = Path(r"I:\Departamentos\Óptica\paulabp\master\TFM\MFBD images\cropped_shifted_originals")
output_base_path = Path(r"I:\Departamentos\Óptica\paulabp\master\TFM\MFBD images\FFT")

image_mapping = {
    ("CS", "binarias"): [
        "CHR181_cropped_shifted.tif", "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted.tif",
        "CHR181_cropped_shifted_average.tif", "wds15245_I_20230510_2000_3s_100p_selected_500_cropped_shifted_average.tif"
    ],
    ("CS", "dudosas-binarias"): [
        "LAB4_cropped_shifted.tif", "STF1967_cropped_shifted.tif", "YSC8AB_cropped_shifted.tif", 
        "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted.tif", "LAB4_cropped_shifted_average.tif", 
        "STF1967_cropped_shifted_average.tif", "YSC8AB_cropped_shifted_average.tif", 
        "hu874_I_20240228_max_3s_100p_selected_500_cropped_shifted_average.tif"
    ],
    ("CS", "simples"): [
        "COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted.tif", "FK384_cropped_shifted.tif", 
        "COU1897_f3_I_20240317_max_3s_100p_selected_600_cropped_shifted_average.tif", "FK384_cropped_shifted_average.tif"
    ],
    ("NOT", "binarias"): [
        "55Uma_NOT_cropped_shifted.tif", "CHR181_NOT_cropped_shifted.tif", "wds16289_NOT_cropped_shifted.tif", 
        "55Uma_NOT_cropped_shifted_average.tif", "CHR181_NOT_cropped_shifted_average.tif", "wds16289_NOT_cropped_shifted_average.tif"
    ],
    ("NOT", "dudosas-binarias"): [
        "wds14514_NOT_cropped_shifted.tif", "wds14514_NOT_cropped_shifted_average.tif"
    ],
    ("NOT", "simples"): [
        "COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted.tif", 
        "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted.tif", 
        "COU1987_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_average.tif", 
        "KUI48_I_20240124_NOT_max_3s_100p_selected_500_cropped_shifted_average.tif"
    ]
}

# --- Loop Execution ---
for (telescope, star), images in image_mapping.items():
    input_folder = input_base_path / telescope / star
    output_folder = output_base_path / telescope / star
    
    for image in images:
        path_image = input_folder / image
        if path_image.exists():
            process_tiff_stack_fft(path_image, output_folder, apodize=True)
        else:
            print(f"File not found: {path_image}")