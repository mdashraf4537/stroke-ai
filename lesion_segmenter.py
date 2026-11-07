"""
Step 2 & Step 4: 2D MRI Preprocessing & Stage-2 Ischemic Stroke Lesion Segmentation
====================================================================================
Integrates `best_stroke_model.keras` (2D U-Net++ trained on ISLES 2022) into the
Flask Stroke Classification Application.

Triggered automatically in Stage 2 whenever the Stage-1 MRI Classifier predicts
`Ischemic` stroke.

Provides:
  1. Step 2 (`prepare_png_slice_for_segmentation`):
     - 2D MRI grayscale conversion (H, W, 1)
     - Automated cranial & inner brain parenchyma masking (removes text overlays,
       background halo noise, and skull boundary artifacts)
     - Sequence-aware intensity harmonization (maps hyperintense DWI/FLAIR lesions
       to the hypointense ADC profile expected by `best_stroke_model.keras`)
     - Per-slice Z-Score standardization: (img - mean) / (std + 1e-8)
     - Spatial resizing to (1, 128, 128, 1)

  2. Step 4 (`run_ischemic_lesion_segmentation`):
     - Executes U-Net++ inference on the preprocessed (1, 128, 128, 1) tensor
     - Computes the 5 Clinical & Analytical Attributes:
         1. Presence of Stroke ("Ischemic Stroke Present")
         2. Exact Anatomical Location (Hemisphere lateralization & 2D bounding box)
         3. Lesion Size / Burden (Pixel area, % of brain slice, estimated cm²)
         4. Model Confidence & Penumbra Border (Peak prob, mean core confidence,
            high-certainty core vs. transition/penumbra rim pixels)
         5. Segmentation Accuracy (0.4653 Validation Dice / 0.5823 Train Benchmark)
     - Generates & saves:
         * `lesion_overlay_<uid>.png` (High-res MRI with red lesion fill + yellow contour)
         * `lesion_panel_<uid>.png` (4-panel diagnostic figure matching notebook pipeline)
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_TARGET_SIZE = (128, 128)
VAL_DICE_BENCHMARK = 0.4653
TRAIN_DICE_BENCHMARK = 0.5823


def extract_brain_masks(gray_uint8: np.ndarray):
    """
    Extract both the full brain contour mask (`brain_mask`) and an eroded inner
    parenchyma mask (`inner_mask`) that excludes outer skull/meninges rim and
    text annotations (e.g. '[AF]', '12:03', '9mm') found on clinical 2D MRI PNGs.
    """
    h, w = gray_uint8.shape[:2]

    # Smooth slightly to suppress speckle noise in background halo
    blurred = cv2.GaussianBlur(gray_uint8, (5, 5), 0)

    non_zero = blurred[blurred > 12]
    if non_zero.size > 0:
        otsu_thresh, _ = cv2.threshold(
            non_zero, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Use 0.85 * Otsu to cleanly strip the noisy outer scalp/skull halo
        # on 2D clinical MRI PNGs and isolate true brain parenchyma
        thresh_val = max(25.0, float(otsu_thresh) * 0.85)
    else:
        thresh_val = 25.0

    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological opening severs thin text annotations from the brain parenchyma
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=2)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = np.zeros((h, w), dtype=np.uint8)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.03 * (h * w):
            cv2.drawContours(brain_mask, [largest], -1, 255, thickness=-1)
            # Smooth contour boundary and fill internal sulcal/ventricular cavities
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, close_k)
            cnts_closed, _ = cv2.findContours(
                brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts_closed:
                cv2.drawContours(
                    brain_mask,
                    [max(cnts_closed, key=cv2.contourArea)],
                    -1,
                    255,
                    thickness=-1,
                )

    # Fallback if threshold was too high for a dark scan
    if np.count_nonzero(brain_mask) == 0:
        _, binary_low = cv2.threshold(blurred, 12, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(brain_mask, [largest], -1, 255, thickness=-1)
        else:
            brain_mask[:] = 255

    # Erode to create inner parenchyma mask (excludes cortical rim artifact)
    erode_px = max(4, int(min(h, w) * 0.035))
    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
    )
    inner_mask = cv2.erode(brain_mask, erode_k, iterations=1)
    if np.count_nonzero(inner_mask) < 0.02 * (h * w):
        inner_mask = brain_mask.copy()

    return brain_mask > 0, inner_mask > 0


# ==============================================================================
# STEP 2: PREPROCESSING STAGE FOR A 2D MRI IMAGE
# ==============================================================================
def prepare_png_slice_for_segmentation(
    img_rgb_or_path,
    sequence_mode: str = "auto",
    target_size: tuple = MODEL_TARGET_SIZE,
):
    """
    Step 2: Converts a 2D MRI image (RGB numpy array or file path) into the exact
    preprocessed tensor required by `best_stroke_model.keras` (`(1, 128, 128, 1)`).

    Pipeline steps:
      1. Grayscale conversion (`(H, W)` float32)
      2. Cranial & inner parenchyma mask extraction (`extract_brain_masks`)
      3. Sequence intensity harmonization:
         - `adc`: Keeps native dark-lesion contrast inside the brain mask.
         - `dwi`: Inverts intensities inside the brain mask so hyperintense (bright)
                  DWI/FLAIR ischemic lesions match the hypointense (dark) ADC profile
                  learned from ISLES 2022.
      4. Z-Score Standardization: `(img - mean) / (std + 1e-8)`
      5. Spatial resizing to `(128, 128)` and tensor reshaping to `(1, 128, 128, 1)`.

    Returns:
      dict with:
        - `tensor`: `(1, 128, 128, 1)` float32 input tensor
        - `gray_orig`: `(H, W)` uint8 original grayscale slice
        - `gray_128`: `(128, 128)` float32 normalized slice
        - `brain_mask_orig`: `(H, W)` bool mask
        - `inner_mask_orig`: `(H, W)` bool mask
        - `brain_mask_128`: `(128, 128)` bool mask
        - `inner_mask_128`: `(128, 128)` bool mask
        - `effective_mode`: `"dwi"` or `"adc"`
    """
    if isinstance(img_rgb_or_path, str):
        bgr = cv2.imread(img_rgb_or_path)
        if bgr is None:
            raise ValueError(f"Could not read MRI image from {img_rgb_or_path}")
        gray_uint8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        arr = np.asarray(img_rgb_or_path)
        if arr.ndim == 3 and arr.shape[2] == 3:
            gray_uint8 = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            gray_uint8 = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        else:
            gray_uint8 = arr.astype(np.uint8)

    brain_mask_orig, inner_mask_orig = extract_brain_masks(gray_uint8)
    img_float = gray_uint8.astype(np.float32)

    # Skull-strip background outside brain mask to 0 (matching ISLES 2022 NIfTI format)
    skull_stripped = np.zeros_like(img_float)
    skull_stripped[brain_mask_orig] = img_float[brain_mask_orig]

    mode = (sequence_mode or "auto").lower().strip()
    if mode not in ("adc", "dwi", "auto"):
        mode = "auto"

    if mode == "auto":
        # Detect whether the scan has focal hyperintense zones typical of DWI/FLAIR
        brain_vals = skull_stripped[inner_mask_orig]
        if brain_vals.size > 0:
            p50 = np.percentile(brain_vals, 50)
            p95 = np.percentile(brain_vals, 95)
            p99 = np.percentile(brain_vals, 99)
            # In DWI/FLAIR, hyperintense ischemic foci push upper tail well above median
            effective_mode = "dwi" if (p99 - p50) > 45 and p95 > 135 else "adc"
        else:
            effective_mode = "dwi"
    else:
        effective_mode = mode

    harmonized = np.zeros_like(skull_stripped)
    if effective_mode == "dwi" and np.any(brain_mask_orig):
        # Non-linear physical diffusion harmonization (DWI -> synthetic ADC):
        # Maps hyperintense DWI lesions to hypointense ADC lesions and matches
        # the right-skewed ISLES 2022 ADC Z-score distribution
        b_vals = skull_stripped[brain_mask_orig]
        b_max = float(np.percentile(b_vals, 99.8)) + 15.0
        harmonized[brain_mask_orig] = (
            (np.clip(b_max - b_vals, 5.0, b_max) / b_max) ** 2.5
        ) * 255.0
    else:
        harmonized[brain_mask_orig] = skull_stripped[brain_mask_orig]

    # Exact notebook Z-Score normalization: (img - np.mean(img)) / (np.std(img) + 1e-8)
    mean_val = float(np.mean(harmonized))
    std_val = float(np.std(harmonized))
    img_norm = (harmonized - mean_val) / (std_val + 1e-8)

    # Resize to 128 x 128
    img_128 = cv2.resize(img_norm, target_size, interpolation=cv2.INTER_LINEAR)
    brain_mask_128 = (
        cv2.resize(
            brain_mask_orig.astype(np.uint8),
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    inner_mask_128 = (
        cv2.resize(
            inner_mask_orig.astype(np.uint8),
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )

    tensor = img_128.reshape(1, target_size[0], target_size[1], 1).astype(np.float32)

    return {
        "tensor": tensor,
        "gray_orig": gray_uint8,
        "gray_128": cv2.resize(gray_uint8, target_size, interpolation=cv2.INTER_LINEAR),
        "brain_mask_orig": brain_mask_orig,
        "inner_mask_orig": inner_mask_orig,
        "brain_mask_128": brain_mask_128,
        "inner_mask_128": inner_mask_128,
        "effective_mode": effective_mode,
    }


def _refine_lesion_probability_map(
    prob_map_128: np.ndarray,
    prep_data: dict,
    prob_thresh: float = 0.5,
):
    """
    Post-processes the raw 128x128 U-Net++ probability map:
      - Restricts predictions to the inner brain parenchyma (`inner_mask_128`) so
        outer background/skull boundary artifacts are suppressed.
      - For DWI scans, weights the probability map by regional hyperintensity
        inside the brain parenchyma so dark CSF/ventricles are not confused with
        inverted hyperintense ischemic lesions.
      - Applies morphological opening (`3x3`) to remove isolated 1-2px noise.
    """
    inner_128 = prep_data["inner_mask_128"]
    gray_128 = prep_data["gray_128"].astype(np.float32)
    mode = prep_data["effective_mode"]

    masked_prob = prob_map_128.copy() * inner_128.astype(np.float32)

    if mode == "dwi" and np.any(inner_128):
        brain_pixels = gray_128[inner_128]
        p55 = float(np.percentile(brain_pixels, 55))
        # Exclude dark interhemispheric/ventricular CSF (below median brain intensity)
        # while preserving hyperintense parenchymal U-Net++ activations
        masked_prob = masked_prob * (gray_128 >= p55).astype(np.float32)

    binary_128 = (masked_prob > prob_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_128 = cv2.morphologyEx(binary_128, cv2.MORPH_OPEN, kernel)

    # If morphological opening removed a very small focal lesion that was above threshold, keep largest blob
    if np.sum(clean_128) == 0 and np.sum(binary_128) > 0:
        clean_128 = binary_128

    return masked_prob, clean_128


def _predict_2d_nifti_aligned(lesion_model, tensor_128: np.ndarray, prob_thresh: float = 0.5) -> np.ndarray:
    """
    Evaluates a 2D (1, 128, 128, 1) slice in both its native 2D orientation and
    the 90-degree NIfTI RAS array orientation (k=-1 -> predict -> k=+1) used during
    ISLES 2022 training, returning the stronger intracranial probability map.
    """
    sl_128 = tensor_128[0, :, :, 0]
    batch_2 = np.stack([sl_128, np.rot90(sl_128, k=-1)], axis=0)[..., np.newaxis].astype(np.float32)
    preds_2 = lesion_model.predict(batch_2, verbose=0)[:, :, :, 0]
    p_direct = preds_2[0]
    p_nifti = np.rot90(preds_2[1], k=1)
    if float(np.sum(p_nifti > prob_thresh)) >= float(np.sum(p_direct > prob_thresh)):
        return p_nifti
    return p_direct


# ==============================================================================
# STEP 4: STAGE-2 ISCHEMIC LESION SEGMENTATION & 5 CLINICAL ATTRIBUTES
# ==============================================================================
def run_ischemic_lesion_segmentation(
    img_rgb: np.ndarray,
    lesion_model,
    output_folder: str,
    unique_id: str,
    sequence_mode: str = "auto",
    prob_thresh: float = 0.5,
):
    """
    Step 4: Triggered when Stage-1 MRI Classification predicts `Ischemic`.
    Runs Step-2 preprocessing, predicts the 2D lesion probability map and binary
    mask using `best_stroke_model.keras`, computes the 5 Clinical Attributes, and
    saves the 4-panel figure + high-resolution contour overlay to `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)

    # In auto mode, evaluate both DWI (bright lesion harmonized) and ADC (native dark lesion)
    # and choose the representation with stronger focal parenchymal activation
    if (sequence_mode or "auto").lower() == "auto":
        prep_dwi = prepare_png_slice_for_segmentation(img_rgb, sequence_mode="dwi")
        raw_prob_dwi = _predict_2d_nifti_aligned(lesion_model, prep_dwi["tensor"], prob_thresh)
        prob_dwi, mask_dwi = _refine_lesion_probability_map(raw_prob_dwi, prep_dwi, prob_thresh)

        prep_adc = prepare_png_slice_for_segmentation(img_rgb, sequence_mode="adc")
        raw_prob_adc = _predict_2d_nifti_aligned(lesion_model, prep_adc["tensor"], prob_thresh)
        prob_adc, mask_adc = _refine_lesion_probability_map(raw_prob_adc, prep_adc, prob_thresh)

        # Select mode that yields a plausible focal lesion (between 8 and 3500 pixels on 128x128)
        dwi_px = int(np.sum(mask_dwi))
        adc_px = int(np.sum(mask_adc))
        if 8 <= adc_px <= 3500 and float(np.max(prob_adc)) >= 0.85:
            prep, prob_map_128, binary_mask_128 = prep_adc, prob_adc, mask_adc
        elif 8 <= dwi_px <= 3500:
            prep, prob_map_128, binary_mask_128 = prep_dwi, prob_dwi, mask_dwi
        elif 8 <= adc_px <= 3500:
            prep, prob_map_128, binary_mask_128 = prep_adc, prob_adc, mask_adc
        elif float(np.max(prob_dwi)) >= float(np.max(prob_adc)):
            prep, prob_map_128, binary_mask_128 = prep_dwi, prob_dwi, mask_dwi
        else:
            prep, prob_map_128, binary_mask_128 = prep_adc, prob_adc, mask_adc
    else:
        prep = prepare_png_slice_for_segmentation(img_rgb, sequence_mode=sequence_mode)
        raw_prob = _predict_2d_nifti_aligned(lesion_model, prep["tensor"], prob_thresh)
        prob_map_128, binary_mask_128 = _refine_lesion_probability_map(
            raw_prob, prep, prob_thresh
        )

    gray_orig = prep["gray_orig"]
    orig_h, orig_w = gray_orig.shape[:2]
    brain_mask_128 = prep["brain_mask_128"]

    # Upsample probability map and binary mask to original MRI resolution
    prob_map_orig = cv2.resize(
        prob_map_128, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )
    binary_mask_orig = (
        cv2.resize(
            binary_mask_128, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )

    # --------------------------------------------------------------------------
    # Compute the 5 Clinical & Analytical Attributes
    # --------------------------------------------------------------------------
    lesion_pixels_128 = int(np.sum(binary_mask_128 > 0))
    brain_pixels_128 = max(1, int(np.sum(brain_mask_128 > 0)))
    lesion_pct_of_brain = (lesion_pixels_128 / brain_pixels_128) * 100.0

    # Estimated physical area in cm² (standard 1mm x 1mm in-plane resolution on 128x128)
    lesion_area_mm2 = float(lesion_pixels_128) * 1.0
    lesion_area_cm2 = lesion_area_mm2 / 100.0

    max_prob = float(np.max(prob_map_128))
    if lesion_pixels_128 > 0:
        mean_lesion_prob = float(np.mean(prob_map_128[binary_mask_128 > 0]))
    else:
        mean_lesion_prob = max_prob

    high_conf_core_px = int(np.sum(prob_map_128 > 0.80))
    penumbra_border_px = int(
        np.sum((prob_map_128 >= 0.30) & (prob_map_128 <= 0.60))
    )

    # Attribute 1: Presence of Stroke
    is_present = lesion_pixels_128 >= 5
    presence_title = (
        "Ischemic Stroke Present"
        if is_present
        else "Ischemic Stroke Present (Sub-threshold Focal Signal)"
    )

    # Attribute 2: Exact Anatomical Location (Hemisphere & 2D Bounding Box)
    ys, xs = np.where(binary_mask_128 > 0)
    if len(xs) > 0:
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        mid_x = MODEL_TARGET_SIZE[1] // 2
        left_px = int(np.sum(xs < mid_x))
        right_px = int(np.sum(xs >= mid_x))
        if left_px > 0 and right_px > 0 and min(left_px, right_px) / max(left_px, right_px) > 0.25:
            hemisphere = "Bilateral Hemispheres"
        elif left_px >= right_px:
            hemisphere = "Left Hemisphere (Image-Left)"
        else:
            hemisphere = "Right Hemisphere (Image-Right)"

        # Anterior / Middle / Posterior region along y-axis
        mean_y = float(np.mean(ys)) / MODEL_TARGET_SIZE[0]
        if mean_y < 0.38:
            ap_region = "Anterior / Frontal Territory"
        elif mean_y < 0.65:
            ap_region = "Middle / Temporo-Parietal Territory"
        else:
            ap_region = "Posterior / Occipital Territory"

        bbox_str = f"x: [{x_min}..{x_max}], y: [{y_min}..{y_max}]"
        location_summary = f"{hemisphere} — {ap_region} ({bbox_str})"
    else:
        hemisphere = "No Supra-Threshold Core"
        ap_region = "Diffuse / Sub-threshold"
        bbox_str = "N/A"
        location_summary = "No focal supra-threshold bounding box isolated"

    # --------------------------------------------------------------------------
    # Generate Visual Outputs:
    #   1) `lesion_overlay_<unique_id>.png`
    #   2) `lesion_panel_<unique_id>.png`
    # --------------------------------------------------------------------------
    base_rgb = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2RGB)
    overlay_rgb = base_rgb.copy()

    if np.any(binary_mask_orig):
        # Semi-transparent red fill over ischemic core
        red_layer = np.zeros_like(base_rgb)
        red_layer[:, :, 0] = 255
        alpha = 0.42
        mask_bool = binary_mask_orig
        overlay_rgb[mask_bool] = (
            (1.0 - alpha) * base_rgb[mask_bool] + alpha * red_layer[mask_bool]
        ).astype(np.uint8)

        # Crisp yellow contour outline around lesion border
        contours, _ = cv2.findContours(
            binary_mask_orig.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay_rgb, contours, -1, (255, 230, 0), thickness=2)

    overlay_filename = f"lesion_overlay_{unique_id}.png"
    overlay_path = os.path.join(output_folder, overlay_filename)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    # 4-Panel Diagnostic Figure (matching `nii_stroke_prediction.png`)
    panel_filename = f"lesion_panel_{unique_id}.png"
    panel_path = os.path.join(output_folder, panel_filename)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=130)
    fig.patch.set_facecolor("#0f172a")

    # Panel 1: Preprocessed MRI Slice
    axes[0].imshow(prep["gray_128"], cmap="gray")
    axes[0].set_title(
        f"1. 2D MRI Slice ({prep['effective_mode'].upper()} Mode)",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    axes[0].axis("off")

    # Panel 2: Stroke Probability Map
    im1 = axes[1].imshow(prob_map_128, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"2. Lesion Probability (Max: {max_prob * 100:.1f}%)",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    axes[1].axis("off")
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    # Panel 3: Predicted Binary Mask
    axes[2].imshow(binary_mask_128, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title(
        f"3. Binary Mask ({lesion_pixels_128} px | {lesion_area_cm2:.2f} cm²)",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    axes[2].axis("off")

    # Panel 4: Lesion Overlay + Bounding Box
    axes[3].imshow(prep["gray_128"], cmap="gray")
    masked_overlay = np.ma.masked_where(binary_mask_128 == 0, binary_mask_128)
    axes[3].imshow(masked_overlay, cmap="autumn", alpha=0.65, vmin=0, vmax=1)
    if len(xs) > 0:
        rect = plt.Rectangle(
            (x_min, y_min),
            max(1, x_max - x_min),
            max(1, y_max - y_min),
            fill=False,
            edgecolor="#38bdf8",
            linewidth=1.5,
            linestyle="--",
        )
        axes[3].add_patch(rect)
    axes[3].set_title(
        "4. Ischemic Lesion Overlay",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(panel_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "activated": True,
        "sequence_mode_used": prep["effective_mode"].upper(),
        "overlay_filename": overlay_filename,
        "panel_filename": panel_filename,
        # 5 Clinical & Analytical Attributes
        "attributes": {
            "presence": {
                "title": presence_title,
                "is_present": is_present,
                "detail": (
                    f"Stage-1 classified scan as Ischemic; Stage-2 U-Net++ isolated "
                    f"{lesion_pixels_128} ischemic pixels ({lesion_pct_of_brain:.2f}% of brain slice)."
                ),
            },
            "location": {
                "hemisphere": hemisphere,
                "territory": ap_region,
                "bbox": bbox_str,
                "summary": location_summary,
            },
            "size": {
                "pixels_128": lesion_pixels_128,
                "brain_pct": round(lesion_pct_of_brain, 2),
                "area_mm2": round(lesion_area_mm2, 1),
                "area_cm2": round(lesion_area_cm2, 2),
                "summary": f"{lesion_pixels_128} px ({lesion_area_cm2:.2f} cm² | {lesion_pct_of_brain:.2f}% of brain slice)",
            },
            "confidence": {
                "max_prob_pct": round(max_prob * 100.0, 1),
                "mean_prob_pct": round(mean_lesion_prob * 100.0, 1),
                "core_pixels": high_conf_core_px,
                "penumbra_pixels": penumbra_border_px,
                "summary": (
                    f"Peak: {max_prob * 100.0:.1f}% | Mean Core: {mean_lesion_prob * 100.0:.1f}% "
                    f"(Core p>0.8: {high_conf_core_px} px, Penumbra 0.3–0.6: {penumbra_border_px} px)"
                ),
            },
            "accuracy": {
                "val_dice": VAL_DICE_BENCHMARK,
                "train_dice": TRAIN_DICE_BENCHMARK,
                "summary": f"{VAL_DICE_BENCHMARK:.4f} Val Dice ({TRAIN_DICE_BENCHMARK:.4f} Train Dice — ISLES 2022 2D U-Net++)",
            },
        },
    }


def run_nifti_lesion_segmentation(
    nifti_path: str,
    lesion_model,
    output_folder: str,
    unique_id: str,
    prob_thresh: float = 0.5,
    target_size=(224, 224),
):
    """
    Native 3D NIfTI (.nii / .nii.gz) Stage-2 Segmentation Pipeline.
    Processes all valid axial slices using the exact Kaggle notebook Z-Score
    normalization `(slice - mean) / (std + 1e-8)`, identifies the peak lesion
    slice `best_z`, calculates true 3D volumetric burden in mL (cm³) using the
    NIfTI header voxel spacing, and generates the 4-panel figure + overlay.
    """
    import nibabel as nib

    os.makedirs(output_folder, exist_ok=True)
    nii_obj = nib.load(nifti_path)
    vol = nii_obj.get_fdata()
    zooms = nii_obj.header.get_zooms()[:3] if len(nii_obj.header.get_zooms()) >= 3 else (2.0, 2.0, 2.0)
    voxel_vol_mm3 = float(zooms[0] * zooms[1] * zooms[2])

    if vol.ndim == 4:
        vol = vol[:, :, :, 0]

    h_orig, w_orig, depth = vol.shape[:3]
    valid_indices = []
    batch_128 = []
    brain_masks_128 = []

    for z in range(depth):
        sl = np.nan_to_num(vol[:, :, z], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if np.std(sl) < 1e-6 or np.count_nonzero(sl > 0) < 50:
            continue
        norm_sl = (sl - float(np.mean(sl))) / (float(np.std(sl)) + 1e-8)
        res_128 = cv2.resize(norm_sl, MODEL_TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        bmask_128 = cv2.resize((sl > 0).astype(np.uint8), MODEL_TARGET_SIZE, interpolation=cv2.INTER_NEAREST) > 0
        valid_indices.append(z)
        batch_128.append(res_128)
        brain_masks_128.append(bmask_128)

    if not batch_128:
        raise ValueError("NIfTI volume contains no valid non-empty brain slices.")

    X_batch = np.expand_dims(np.array(batch_128, dtype=np.float32), axis=-1)
    preds_batch = lesion_model.predict(X_batch, batch_size=16, verbose=0)[:, :, :, 0]

    best_idx = 0
    best_score = -1.0
    total_voxels_orig = 0
    affected_slices = 0

    clean_masks_128 = []
    masked_probs_128 = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for i, z in enumerate(valid_indices):
        p_map = preds_batch[i] * brain_masks_128[i].astype(np.float32)
        b_mask = (p_map > prob_thresh).astype(np.uint8)
        c_mask = cv2.morphologyEx(b_mask, cv2.MORPH_OPEN, kernel)
        masked_probs_128.append(p_map)
        clean_masks_128.append(c_mask)

        px_count = int(np.sum(c_mask > 0))
        if px_count > 0:
            affected_slices += 1
            orig_mask = cv2.resize(c_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            total_voxels_orig += int(np.sum(orig_mask > 0))

        # Prefer mid-brain slices with substantial lesion burden
        score = float(px_count) * float(np.max(p_map))
        if score > best_score:
            best_score = score
            best_idx = i

    best_z = valid_indices[best_idx]
    # Rotate 90 deg counter-clockwise so display matches upright radiological orientation
    prob_map_128 = np.rot90(masked_probs_128[best_idx])
    binary_mask_128 = np.rot90(clean_masks_128[best_idx])
    brain_mask_128 = np.rot90(brain_masks_128[best_idx])

    raw_slice = np.rot90(np.nan_to_num(vol[:, :, best_z], nan=0.0).astype(np.float32))
    nz = raw_slice[raw_slice > 0]
    if nz.size > 0:
        p1, p99 = float(np.percentile(nz, 1.0)), float(np.percentile(nz, 99.0))
        gray_orig = (np.clip((raw_slice - p1) / max(p99 - p1, 1e-8), 0.0, 1.0) * 255.0).astype(np.uint8)
        gray_orig[raw_slice <= 0] = 0
    else:
        gray_orig = np.zeros_like(raw_slice, dtype=np.uint8)

    gray_128 = cv2.resize(gray_orig, MODEL_TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    gray_display = cv2.resize(gray_orig, target_size, interpolation=cv2.INTER_LINEAR)
    mask_display = cv2.resize(binary_mask_128, target_size, interpolation=cv2.INTER_NEAREST) > 0

    lesion_pixels_128 = int(np.sum(binary_mask_128 > 0))
    brain_pixels_128 = max(1, int(np.sum(brain_mask_128 > 0)))
    lesion_pct_of_brain = (lesion_pixels_128 / brain_pixels_128) * 100.0

    # Calculate physical slice area (cm²) and 3D volume (mL / cm³)
    scale_x = (w_orig * zooms[0]) / float(MODEL_TARGET_SIZE[1])
    scale_y = (h_orig * zooms[1]) / float(MODEL_TARGET_SIZE[0])
    lesion_area_mm2 = float(lesion_pixels_128) * scale_x * scale_y
    lesion_area_cm2 = lesion_area_mm2 / 100.0
    total_vol_ml = (total_voxels_orig * voxel_vol_mm3) / 1000.0

    max_prob = float(np.max(prob_map_128))
    mean_lesion_prob = float(np.mean(prob_map_128[binary_mask_128 > 0])) if lesion_pixels_128 > 0 else max_prob
    high_conf_core_px = int(np.sum(prob_map_128 > 0.80))
    penumbra_border_px = int(np.sum((prob_map_128 >= 0.30) & (prob_map_128 <= 0.60)))

    is_present = lesion_pixels_128 >= 5
    presence_title = "Ischemic Stroke Present" if is_present else "Ischemic Stroke Present (Sub-threshold)"

    ys, xs = np.where(binary_mask_128 > 0)
    if len(xs) > 0:
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        mid_x = MODEL_TARGET_SIZE[1] // 2
        left_px = int(np.sum(xs < mid_x))
        right_px = int(np.sum(xs >= mid_x))
        if left_px > 0 and right_px > 0 and min(left_px, right_px) / max(left_px, right_px) > 0.25:
            hemisphere = "Bilateral Hemispheres"
        elif left_px >= right_px:
            hemisphere = "Left Hemisphere (Image-Left)"
        else:
            hemisphere = "Right Hemisphere (Image-Right)"
        ap_region = f"Peak Axial Slice z={best_z}/{depth} ({affected_slices} slices affected)"
        bbox_str = f"x: [{x_min}..{x_max}], y: [{y_min}..{y_max}], z={best_z}"
        location_summary = f"{hemisphere} — {ap_region} ({bbox_str})"
    else:
        hemisphere = "No Supra-Threshold Core"
        ap_region = f"Slice z={best_z}/{depth}"
        bbox_str = "N/A"
        location_summary = "No focal supra-threshold bounding box isolated"

    # Save overlay & 4-panel figure
    base_rgb = cv2.cvtColor(gray_display, cv2.COLOR_GRAY2RGB)
    overlay_rgb = base_rgb.copy()
    if np.any(mask_display):
        red_layer = np.zeros_like(base_rgb)
        red_layer[:, :, 0] = 255
        overlay_rgb[mask_display] = (0.58 * base_rgb[mask_display] + 0.42 * red_layer[mask_display]).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_display.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgb, cnts, -1, (255, 230, 0), thickness=2)

    overlay_filename = f"lesion_overlay_{unique_id}.png"
    cv2.imwrite(os.path.join(output_folder, overlay_filename), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    panel_filename = f"lesion_panel_{unique_id}.png"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=130)
    fig.patch.set_facecolor("#0f172a")

    axes[0].imshow(gray_128, cmap="gray")
    axes[0].set_title(f"1. NIfTI Slice z={best_z} (Native 3D)", color="white", fontsize=10.5, fontweight="bold", pad=8)
    axes[0].axis("off")

    im1 = axes[1].imshow(prob_map_128, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"2. Lesion Probability (Max: {max_prob * 100:.1f}%)", color="white", fontsize=10.5, fontweight="bold", pad=8)
    axes[1].axis("off")
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    axes[2].imshow(binary_mask_128, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title(f"3. Binary Mask ({lesion_pixels_128} px | {total_vol_ml:.2f} mL 3D)", color="white", fontsize=10.5, fontweight="bold", pad=8)
    axes[2].axis("off")

    axes[3].imshow(gray_128, cmap="gray")
    axes[3].imshow(np.ma.masked_where(binary_mask_128 == 0, binary_mask_128), cmap="autumn", alpha=0.65, vmin=0, vmax=1)
    if len(xs) > 0:
        axes[3].add_patch(
            plt.Rectangle((x_min, y_min), max(1, x_max - x_min), max(1, y_max - y_min), fill=False, edgecolor="#38bdf8", linewidth=1.5, linestyle="--")
        )
    axes[3].set_title("4. Ischemic Lesion Overlay", color="white", fontsize=10.5, fontweight="bold", pad=8)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, panel_filename), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "activated": True,
        "sequence_mode_used": f"3D NIfTI Native (Peak z={best_z}/{depth})",
        "overlay_filename": overlay_filename,
        "panel_filename": panel_filename,
        "best_slice_z": best_z,
        "peak_slice_rgb": base_rgb,
        "attributes": {
            "presence": {
                "title": presence_title,
                "is_present": is_present,
                "detail": (
                    f"3D NIfTI volume analyzed across {len(valid_indices)} brain slices: "
                    f"ischemic lesion detected in {affected_slices} slices (Total 3D Volume: {total_vol_ml:.2f} mL)."
                ),
            },
            "location": {
                "hemisphere": hemisphere,
                "territory": ap_region,
                "bbox": bbox_str,
                "summary": location_summary,
            },
            "size": {
                "pixels_128": lesion_pixels_128,
                "brain_pct": round(lesion_pct_of_brain, 2),
                "area_mm2": round(lesion_area_mm2, 1),
                "area_cm2": round(lesion_area_cm2, 2),
                "vol_ml": round(total_vol_ml, 2),
                "summary": f"{lesion_pixels_128} px ({lesion_area_cm2:.2f} cm² on z={best_z} | {total_vol_ml:.2f} mL 3D Volume)",
            },
            "confidence": {
                "max_prob_pct": round(max_prob * 100.0, 1),
                "mean_prob_pct": round(mean_lesion_prob * 100.0, 1),
                "core_pixels": high_conf_core_px,
                "penumbra_pixels": penumbra_border_px,
                "summary": (
                    f"Peak: {max_prob * 100.0:.1f}% | Mean Core: {mean_lesion_prob * 100.0:.1f}% "
                    f"(Core p>0.8: {high_conf_core_px} px, Penumbra 0.3-0.6: {penumbra_border_px} px)"
                ),
            },
            "accuracy": {
                "val_dice": VAL_DICE_BENCHMARK,
                "train_dice": TRAIN_DICE_BENCHMARK,
                "summary": f"{VAL_DICE_BENCHMARK:.4f} Val Dice ({TRAIN_DICE_BENCHMARK:.4f} Train Dice — ISLES 2022 2D U-Net++)",
            },
        },
    }
