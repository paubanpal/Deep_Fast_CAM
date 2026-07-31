from itertools import combinations
from pathlib import Path
import numpy as np
import tifffile as tiff


def load_tiff_images_from_directory(
    dir_path: str | Path,
    extensions: tuple[str, ...] = (".tiff", ".tif"),
) -> tuple[list[np.ndarray], list[Path]]:
    """Finds and loads all TIFF images in a directory using pathlib.Path."""
    directory = Path(dir_path)

    if not directory.exists() or not directory.is_dir():
        msg = f"Directory does not exist or is not a valid directory: {directory}"
        raise NotADirectoryError(msg)

    # Search for files matching the allowed extensions (case-insensitive)
    image_paths = sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
    )

    if not image_paths:
        msg = f"No TIFF images found in {directory} matching extensions {extensions}"
        raise FileNotFoundError(msg)

    # Read images and convert to float32
    images = []
    for path in image_paths:
        try:
            img = tiff.imread(path)
            images.append(img.astype(np.float32))
        except Exception as e:
            msg = f"Failed to read image at {path}: {e}"
            raise IOError(msg) from e

    return images, image_paths


def normalize_images(
    images: list[np.ndarray],
) -> tuple[list[np.ndarray], float]:
    """Normalizes all images relative to the single brightest pixel across the entire dataset."""
    global_max = max(np.max(img) for img in images)

    if global_max == 0:
        msg = "All images are completely black."
        raise ValueError(msg)

    normalized_images = [img / global_max for img in images]
    return normalized_images, float(global_max)


def compare_image_pair(
    img1: np.ndarray, img2: np.ndarray
) -> dict[str, float]:
    """Calculates pixel-by-pixel metrics between two images."""
    if img1.shape != img2.shape:
        msg = f"Image dimensions must match! Got {img1.shape} and {img2.shape}"
        raise ValueError(msg)

    # Matrix subtraction
    diff = img1 - img2

    # Metric calculations
    net_sum_diff = np.sum(diff)  # Signed sum: accounts for direction (+/-)
    abs_sum_diff = np.sum(np.abs(diff))  # Absolute total magnitude of difference
    rms_value = np.sqrt(np.mean(diff**2))  # Root Mean Square

    return {
        "net_sum_diff": float(net_sum_diff),
        "abs_sum_diff": float(abs_sum_diff),
        "rms": float(rms_value),
    }


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Specify your target directory using Path
    input_dir = Path("./my_tiff_folder")

    # 1. Load all TIFF images found in the directory
    raw_images, file_paths = load_tiff_images_from_directory(input_dir)
    print(f"Loaded {len(raw_images)} TIFF files from: {input_dir.resolve()}\n")

    # 2. Global Peak Normalization
    norm_images, global_max_val = normalize_images(raw_images)
    print(
        f"Global peak intensity: {global_max_val} (all images normalized relative to this)\n"
    )

    # 3. Pairwise Comparison
    for i, j in combinations(range(len(norm_images)), 2):
        path_a = file_paths[i].name
        path_b = file_paths[j].name

        results = compare_image_pair(norm_images[i], norm_images[j])

        print(f"--- Comparison: [{path_a}] vs [{path_b}] ---")
        print(f"  Net Sum of Differences (A - B) : {results['net_sum_diff']:+.6f}")
        print(f"  Absolute Sum of Differences    : {results['abs_sum_diff']:.6f}")
        print(f"  RMS Difference                 : {results['rms']:.6f}\n")