import os
import re
import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import nibabel as nib
except ImportError:
    nib = None


def _is_nifti_file(file_path: str) -> bool:
    """Check if the file is a NIfTI (.nii or .nii.gz) neuroimaging volume."""
    lower = file_path.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def load_nifti_slice_rgb(file_path: str, slice_index: int = None) -> np.ndarray:
    """
    Loads a 3D/4D NIfTI (.nii / .nii.gz) volume and extracts a 2D axial slice
    formatted as an 8-bit 3-channel RGB image for Stage-1 2D CNN classification
    and browser display.

    Handles:
      - 4D -> 3D volume squeezing
      - Automatic mid-ventricular axial slice selection (around 58% depth, e.g. z=42/72)
      - 90-degree RAS-to-radiological upright rotation (`np.rot90`)
      - Floating-point physical unit (e.g. 0.0004..0.0035 mm²/s ADC) percentile
        rescaling to [0..255] uint8
    """
    if nib is None:
        raise ImportError(
            "nibabel is required to read .nii / .nii.gz files. Install with 'pip install nibabel'."
        )
    vol = nib.load(file_path).get_fdata()
    if vol.ndim == 4:
        vol = vol[:, :, :, 0]
    if vol.ndim == 2:
        slice_2d = vol
    else:
        depth = vol.shape[2]
        if slice_index is not None and 0 <= slice_index < depth:
            z = slice_index
        else:
            # Select the mid-axial slice (50% depth, e.g. z=36/72) where
            # Stage-1 2D MRI CNN has the best anatomical alignment
            target_z = int(depth * 0.50)
            if 0 <= target_z < depth and np.count_nonzero(vol[:, :, target_z] > 0) > 200:
                z = target_z
            else:
                z = max(range(depth), key=lambda i: int(np.count_nonzero(vol[:, :, i] > 0)))
        slice_2d = vol[:, :, z]

    slice_2d = np.nan_to_num(slice_2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # Rotate 90 degrees counter-clockwise from NIfTI (X, Y) to upright display orientation
    slice_upright = np.rot90(slice_2d)

    non_zero = slice_upright[slice_upright > 0]
    if non_zero.size > 0:
        p99 = float(np.percentile(non_zero, 99.0))
        scaled = (np.clip(slice_upright, 0.0, p99) / (p99 + 1e-8)) * 255.0
        gray_u8 = scaled.astype(np.uint8)
    else:
        gray_u8 = np.zeros_like(slice_upright, dtype=np.uint8)

    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)


def load_raw_image(file_path: str) -> np.ndarray:
    """
    Loads an image from disk. Supports standard image formats (PNG, JPG, JPEG),
    medical DICOM (.dcm) files, and 3D NIfTI (.nii / .nii.gz) volumes.
    Returns:
        RGB numpy array of shape (H, W, 3) with dtype uint8 (0..255).
    """
    ext = os.path.splitext(file_path)[1].lower()

    # 1. Handle NIfTI (.nii / .nii.gz) 3D MRI volume
    if _is_nifti_file(file_path):
        return load_nifti_slice_rgb(file_path)

    # 2. Handle DICOM format if file has .dcm extension or DICOM header
    if ext == ".dcm" or _is_dicom_file(file_path):
        if pydicom is None:
            raise ImportError("pydicom is required to read .dcm medical files. Install with 'pip install pydicom'.")
        return load_dicom_with_brain_window(file_path)

    # 3. Handle standard image formats (PNG, JPG, JPEG, etc.)
    with Image.open(file_path) as pil_img:
        # Correct orientation based on EXIF tags (e.g. smartphone photos)
        pil_img = ImageOps.exif_transpose(pil_img)
        img_rgb = np.array(pil_img.convert("RGB"))
        return img_rgb


def _is_dicom_file(file_path: str) -> bool:
    """Check for standard DICM magic bytes at offset 128."""
    try:
        with open(file_path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


def load_dicom_with_brain_window(file_path: str, window_center: float = 40.0, window_width: float = 80.0) -> np.ndarray:
    """
    Reads a DICOM scan and converts raw pixel values to Hounsfield Units (HU),
    applying the standard radiologist Brain Window:
      - Window Center (WL) = 40 HU
      - Window Width (WW) = 80 HU (range [0, 80] HU)
    Captures acute ischemia (subtle hypodensity) and acute hemorrhage (hyperdense blood).
    """
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixel_array * slope + intercept

    # If the DICOM file has embedded window settings, prefer brain window default for stroke
    img_min = window_center - (window_width / 2.0)
    img_max = window_center + (window_width / 2.0)
    
    windowed = np.clip(hu, img_min, img_max)
    normalized = ((windowed - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    
    # Return 3-channel RGB image
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)


def clean_and_crop_head(img_rgb: np.ndarray, margin_percent: float = 0.03) -> np.ndarray:
    """
    Detects the patient's head/brain boundary and crops out peripheral scanner artifacts,
    black background margins, and patient/clinic text annotations outside the scan.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Threshold background noise (near-black pixels)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_rgb

    # Find the largest contour (the head structure)
    largest_contour = max(contours, key=cv2.contourArea)
    total_area = img_rgb.shape[0] * img_rgb.shape[1]
    contour_area = cv2.contourArea(largest_contour)

    # Only crop if the detected region occupies a reasonable fraction of the frame (>15%)
    if contour_area < (total_area * 0.15):
        return img_rgb

    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add small padding margin
    pad_x = int(w * margin_percent)
    pad_y = int(h * margin_percent)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_rgb.shape[1], x + w + pad_x)
    y2 = min(img_rgb.shape[0], y + h + pad_y)

    cropped = img_rgb[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else img_rgb


def enhance_brain_contrast(img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the luminance
    channel, making faint ischemic infarcts and tissue differences visible across
    diverse scanner brightness calibrations.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def letterbox_resize(img_rgb: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Resizes the image to target_size maintaining the original aspect ratio,
    padding remaining borders with black. Avoids distorting brain proportions.
    """
    target_h, target_w = target_size
    h, w = img_rgb.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def preprocess_real_world_scan(file_path: str, target_size=(224, 224), apply_contrast_enhancement: bool = True):
    """
    Full real-world preprocessing pipeline:
      1. Load raw image / DICOM
      2. Clean peripheral artifacts & crop to head
      3. Enhance contrast (CLAHE)
      4. Aspect-ratio preserving letterbox resize
    Returns:
        preprocessed_rgb: uint8 array (target_size[0], target_size[1], 3)
        raw_rgb: original uint8 array for display
    """
    raw_rgb = load_raw_image(file_path)
    if _is_nifti_file(file_path):
        # Preserve native NIfTI field-of-view margin for 2D CNN compatibility
        final_rgb = cv2.resize(raw_rgb, target_size)
        return final_rgb, raw_rgb

    cropped = clean_and_crop_head(raw_rgb)
    
    if apply_contrast_enhancement:
        enhanced = enhance_brain_contrast(cropped)
    else:
        enhanced = cropped

    final_rgb = letterbox_resize(enhanced, target_size=target_size)
    return final_rgb, raw_rgb


def check_scan_quality(img_rgb: np.ndarray) -> dict:
    """
    Basic sanity check to warn users if the uploaded image has quality issues
    (e.g. completely black, pure white, or excessively blurry).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    std_dev = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    issues = []
    if std_dev < 12.0:
        issues.append("Image has very low contrast or is nearly uniform/empty.")
    if laplacian_var < 15.0:
        issues.append("Image appears significantly blurred or low resolution.")

    return {
        "is_valid": len(issues) == 0,
        "warnings": issues,
        "std_dev": std_dev,
        "sharpness": laplacian_var
    }


def validate_brain_scan(
    file_path: str,
    raw_rgb: np.ndarray,
    expected_modality: str = "ct",
    original_filename: str = "",
) -> dict:
    """
    Validates that the uploaded image is a genuine Brain CT or Brain MRI scan
    and matches the selected diagnostic modality ('ct' or 'mri').
    Rejects natural photographs, colored images, documents, non-cranial images,
    and cross-modality mismatches (CT uploaded to MRI or MRI uploaded to CT).
    """
    modality_label = "CT" if expected_modality.lower() == "ct" else "MRI"
    other_label = "MRI" if modality_label == "CT" else "CT"
    fname_lower = (original_filename or os.path.basename(file_path)).lower()

    # 0. Explicit NIfTI (.nii / .nii.gz) neuroimaging check
    if _is_nifti_file(file_path):
        if expected_modality.lower() == "ct" and any(
            tok in fname_lower for tok in ("mri", "adc", "dwi", "flair", "t1", "t2", "strokeperf")
        ):
            return {
                "is_valid": False,
                "error": "Modality mismatch: Uploaded NIfTI volume is a Brain MRI scan. Please upload a Brain CT scan.",
            }
        gray_nii = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
        if float(np.std(gray_nii)) < 8.0 or int(np.count_nonzero(gray_nii > 10)) < 100:
            return {
                "is_valid": False,
                "error": "Invalid NIfTI volume: Volume appears empty or lacks brain tissue signal.",
            }
        return {"is_valid": True, "error": None}

    # 1. Explicit DICOM header modality check (.dcm)
    ext = os.path.splitext(file_path)[1].lower()
    if (ext == ".dcm" or _is_dicom_file(file_path)) and pydicom is not None:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            dcm_mod = str(getattr(ds, "Modality", "")).upper().strip()
            if expected_modality.lower() == "ct" and dcm_mod == "MR":
                return {
                    "is_valid": False,
                    "error": "Modality mismatch: Uploaded DICOM file is a Brain MRI scan. Please upload a Brain CT scan.",
                }
            if expected_modality.lower() == "mri" and dcm_mod == "CT":
                return {
                    "is_valid": False,
                    "error": "Modality mismatch: Uploaded DICOM file is a Brain CT scan. Please upload a Brain MRI scan.",
                }
        except Exception:
            pass

    # 2. Explicit filename token check for CT vs MRI mismatch
    if expected_modality.lower() == "ct":
        if any(tok in fname_lower for tok in ("mri", "flair", "dwi", "t1w", "t2w")):
            return {
                "is_valid": False,
                "error": "Modality mismatch: Uploaded file appears to be an MRI scan. Please upload a Brain CT scan.",
            }
    elif expected_modality.lower() == "mri":
        if any(tok in fname_lower for tok in ("ct_scan", "ct-scan", "ncct")) or re.search(
            r"(?:^|[_\-\s])ct(?:$|[_\-\s\d])", fname_lower
        ):
            return {
                "is_valid": False,
                "error": "Modality mismatch: Uploaded file appears to be a CT scan. Please upload a Brain MRI scan.",
            }

    # 3. Color / Chromaticity check (Brain CT and MRI scans are grayscale)
    img_f = raw_rgb.astype(np.float32)
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    max_diff = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    colored_pixel_ratio = float(np.mean(max_diff > 25.0))
    mean_color_diff = float(np.mean(max_diff))

    if colored_pixel_ratio > 0.06 or mean_color_diff > 10.0:
        return {
            "is_valid": False,
            "error": f"Invalid image: Color photograph or non-radiological image detected. Only grayscale Brain {modality_label} scans are accepted.",
        }

    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    if h < 64 or w < 64:
        return {
            "is_valid": False,
            "error": f"Invalid image: Resolution is too low for a clinical Brain {modality_label} scan.",
        }

    # 4. Contrast & Blank/Uniformity check
    std_dev = float(np.std(gray))
    if std_dev < 14.0:
        return {
            "is_valid": False,
            "error": f"Invalid image: Uploaded file is blank or lacks anatomical tissue contrast.",
        }

    # 5. Dark-field scanner background check at outer corners
    ch, cw = max(4, int(h * 0.09)), max(4, int(w * 0.09))
    corners = np.concatenate(
        [
            gray[:ch, :cw].ravel(),
            gray[:ch, -cw:].ravel(),
            gray[-ch:, :cw].ravel(),
            gray[-ch:, -cw:].ravel(),
        ]
    )
    corner_median = float(np.median(corners))
    bright_corner_ratio = float(np.mean(corners > 135))
    if corner_median > 80.0 or bright_corner_ratio > 0.35:
        return {
            "is_valid": False,
            "error": f"Invalid image: Missing radiological dark-field scanner background. Please upload a valid Brain {modality_label} scan.",
        }

    # 6. Cranial Vault Contour, Elliptical Solidity & Hemispheric Symmetry check
    _, thresh = cv2.threshold(gray, 24, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "is_valid": False,
            "error": f"Invalid image: No cranial anatomy detected. Please upload a valid Brain {modality_label} scan.",
        }

    largest = max(contours, key=cv2.contourArea)
    c_area = float(cv2.contourArea(largest))
    area_ratio = c_area / float(h * w)
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity = c_area / (hull_area + 1e-6)
    x, y, bw, bh = cv2.boundingRect(largest)
    aspect_ratio = float(bw) / float(bh + 1e-6)

    if area_ratio < 0.12 or area_ratio > 0.93 or not (0.55 <= aspect_ratio <= 1.65) or solidity < 0.70:
        return {
            "is_valid": False,
            "error": f"Invalid image: Foreground geometry does not match cranial brain anatomy. Please upload a valid Brain {modality_label} scan.",
        }

    roi = cv2.resize(gray[y : y + bh, x : x + bw], (200, 200))
    roi_mirror = np.fliplr(roi)
    roi_f = roi.astype(np.float32) - np.mean(roi)
    mir_f = roi_mirror.astype(np.float32) - np.mean(roi_mirror)
    denom = float(np.sqrt(np.sum(roi_f**2) * np.sum(mir_f**2))) + 1e-6
    sym_corr = float(np.sum(roi_f * mir_f) / denom)

    if sym_corr < 0.38:
        return {
            "is_valid": False,
            "error": f"Invalid image: Image lacks bilateral cranial symmetry required for a Brain {modality_label} scan.",
        }

    # 7. Radiographic CT vs MRI Physics Check (Calcified Skull Ring vs Soft-Tissue Profile)
    roi_mask = (roi > 25).astype(np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    eroded_once = cv2.erode(roi_mask, kernel, iterations=1)
    eroded_twice = cv2.erode(roi_mask, kernel, iterations=2)
    rim_mask = (roi_mask - eroded_once) > 0
    inner_mask = eroded_twice > 0

    if np.any(rim_mask) and np.any(inner_mask):
        rim_pixels = roi[rim_mask]
        inner_pixels = roi[inner_mask]
        rim_bright_ratio = float(np.mean(rim_pixels >= 225))
        rim_to_inner_ratio = float(np.mean(rim_pixels)) / (float(np.mean(inner_pixels)) + 1e-5)
        inner_std = float(np.std(inner_pixels))

        # CT scans have a hyperdense calcified skull rim; MRI scans lack calcified bone signal
        if expected_modality.lower() == "ct":
            if rim_bright_ratio < 0.04 and rim_to_inner_ratio < 0.90 and inner_std > 46.0:
                return {
                    "is_valid": False,
                    "error": f"Modality mismatch: Uploaded scan lacks a CT calcified skull ring and matches a Brain {other_label} profile. Please upload a Brain CT scan.",
                }
        elif expected_modality.lower() == "mri":
            if rim_bright_ratio > 0.30 and rim_to_inner_ratio > 1.35 and inner_std < 38.0:
                return {
                    "is_valid": False,
                    "error": f"Modality mismatch: Uploaded scan exhibits a hyperdense calcified skull ring characteristic of a Brain {other_label} scan. Please upload a Brain MRI scan.",
                }

    return {"is_valid": True, "error": None}


def prepare_model_tensor(model, img_rgb: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Prepares batch tensor for prediction without double-rescaling.
    If the model contains a Rescaling layer, keeps 0..255.
    Otherwise normalizes to 0..1 float32.
    """
    import tensorflow as tf
    
    def has_rescaling_layer(m):
        for layer in m.layers:
            if isinstance(layer, tf.keras.layers.Rescaling):
                return True
            if hasattr(layer, "layers"):
                for sub in layer.layers:
                    if isinstance(sub, tf.keras.layers.Rescaling):
                        return True
        return False

    img_float = img_rgb.astype(np.float32)

    if has_rescaling_layer(model):
        x = img_float
    else:
        x = img_float / 255.0

    return np.expand_dims(x, axis=0)
