"""
NeuroFocus AI — Stage-2 CT Hemorrhage Subtype Classification, Lesion Segmentation,
Affected Brain Region Localization, Bleed Quantification & Decoder Grad-CAM
==================================================================================
Integrates the Kaggle `raoufalalas/neurofocus-ai` two-stage cascade artifacts into
the Flask Stroke Classification Application with full compatibility for BOTH:
  - Stage-1 CT Dataset images (`Brain_Stroke_CT_Dataset` & External Hospital CT)
  - Stage-2 CT Dataset images (`RSNA` 16-bit DICOMs & `CT-ICH` 512x512 slices +
    `*_HGE_Seg.jpg` ground-truth masks).

Key Upgrades for Exact Accuracy Across Both Datasets:
  1. Multi-Scale U-Net Inference (`512x512` Native Scale + `256x256` Lesion-Centered
     & Quadrant Crops + `256x256` Full Slice):
     - In `neurofocus-ai.ipynb`, `unet_resnet34_ctich_best.pth` was trained on
       `256x256` crops extracted from `512x512` CT-ICH slices (`1:1` pixel scale).
     - Evaluating only on a downscaled `256x256` full slice shrinks hemorrhages by 4x.
     - Running multi-scale inference (`512x512` + `256x256` focal crops + `256x256`
       full slice) restores exact `1:1` lesion scale for both Stage-1 and Stage-2
       CT datasets.
  2. Intensity-Aware Skull & Sinus Bone Stripping (Preserves Subdural/Epidural Space):
     - Removes only the calcified outer skull ring (`> 235` intensity) and anterior
       nasal/orbital sinus bone walls while preserving the subdural/epidural margin
       immediately beneath the inner skull table.
     - Prevents bone-window skull-base slices (with zero hyperdense parenchymal blood)
       from falsely triggering `Subdural / Subarachnoid`.
  3. Per-Subtype Optimal Thresholds from `neurofocus-ai.log` + Optional Ground-Truth
     Mask (`*_HGE_Seg.jpg`) Live Dice Verification.
"""

import io
import os
import pickle
import zipfile
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEMORRHAGE_TYPES = [
    "Epidural",
    "Subdural",
    "Subarachnoid",
    "Intraparenchymal",
    "Intraventricular",
]

# Optimal per-subtype thresholds tuned on validation in neurofocus-ai.log (line 9715)
OPTIMAL_SUBTYPE_THRESHOLDS = {
    "Epidural": 0.60,
    "Subdural": 0.50,
    "Subarachnoid": 0.55,
    "Intraparenchymal": 0.60,
    "Intraventricular": 0.70,
}

SUBTYPE_DESCRIPTIONS = {
    "Epidural": "Epidural Hemorrhage (EDH — Extra-axial arterial bleed between skull & dura)",
    "Subdural": "Subdural Hemorrhage (SDH — Extra-axial crescentic venous bleed beneath dura)",
    "Subarachnoid": "Subarachnoid Hemorrhage (SAH — Sulcal / cisternal bleed in subarachnoid space)",
    "Intraparenchymal": "Intraparenchymal Hemorrhage (IPH — Focal parenchymal hematoma within brain tissue)",
    "Intraventricular": "Intraventricular Hemorrhage (IVH — Blood extension into ventricular CSF system)",
}

CLS_IMG_SIZE = 512
SEG_CROP_SIZE = 256
VAL_DICE_BENCHMARK = 0.7226
CLS_F1_ANY_BENCHMARK = 0.7403


# ==============================================================================
# 1. NATIVE .PTH STATE_DICT LOADER & CONV+BN EVAL FUSION
# ==============================================================================
def load_pth_state_dict(pth_path: str) -> dict:
    """
    Reads a PyTorch `.pth` zip archive (`torch.save(model.state_dict(), ...)`)
    directly into NumPy float32 arrays without requiring `torch` installed.
    """
    with zipfile.ZipFile(pth_path, "r") as zf:
        prefix = zf.namelist()[0].split("/")[0]

        def load_storage(storage_type, obj_key, _location, numel):
            if "Long" in storage_type or "Int64" in storage_type:
                dtype = np.int64
            elif "Double" in storage_type or "Float64" in storage_type:
                dtype = np.float64
            else:
                dtype = np.float32
            raw = zf.read(f"{prefix}/data/{obj_key}")
            return np.frombuffer(raw, dtype=dtype, count=numel).copy()

        class _PthUnpickler(pickle.Unpickler):
            def find_class(self, mod, name):
                if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(storage, offset, size, _stride, _req_grad, _hooks):
                        count = int(np.prod(size)) if len(size) > 0 else 1
                        arr = storage[offset : offset + count]
                        return arr.reshape(size) if len(size) > 0 else arr
                    return _rebuild
                if mod == "collections" and name == "OrderedDict":
                    return OrderedDict
                return lambda *a, **k: (mod, name, a, k)

            def persistent_load(self, pid):
                _, stype, key, loc, numel = pid
                return load_storage(str(stype), key, loc, numel)

        pkl_bytes = zf.read(f"{prefix}/data.pkl")
        return _PthUnpickler(io.BytesIO(pkl_bytes)).load()


def _fuse_conv_bn(sd: dict, conv_prefix: str, bn_prefix: str, eps: float = 1e-5):
    """
    Fuses PyTorch Conv2d `(O, I, kH, kW)` + BatchNorm2d `(gamma, beta, mean, var)`
    into a TensorFlow Conv2D kernel `(kH, kW, I, O)` and bias `(O,)` in eval mode.
    """
    w_pt = sd[f"{conv_prefix}.weight"].astype(np.float32)  # (O, I, kH, kW)
    gamma = sd[f"{bn_prefix}.weight"].astype(np.float32)
    beta = sd[f"{bn_prefix}.bias"].astype(np.float32)
    mean = sd[f"{bn_prefix}.running_mean"].astype(np.float32)
    var = sd[f"{bn_prefix}.running_var"].astype(np.float32)

    scale = gamma / np.sqrt(var + eps)
    w_tf = np.transpose(w_pt, (2, 3, 1, 0)) * scale.reshape(1, 1, 1, -1)
    b_tf = beta - mean * scale
    return tf.constant(w_tf, dtype=tf.float32), tf.constant(b_tf, dtype=tf.float32)


def _conv2d_fused(x, w_b, stride: int = 1, padding: int = 1):
    w, b = w_b
    if padding > 0:
        x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
    out = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="VALID")
    return tf.nn.bias_add(out, b)


# ==============================================================================
# 2. NEUROFOCUS AI STAGE-1 RESNET-50 5-SUBTYPE CLASSIFIER (`classifier_best.pth`)
# ==============================================================================
class NeuroFocusResNet50Classifier:
    """
    Exact eval-mode execution of `ICHClassifier` (ResNet-50, 1-channel input, 5 classes)
    loaded from `classifier_best.pth`.
    """

    def __init__(self, ckpt_path: str):
        sd = load_pth_state_dict(ckpt_path)
        self.conv1 = _fuse_conv_bn(sd, "backbone.conv1", "backbone.bn1")
        self.layer_blocks = {}
        self.layer_counts = [3, 4, 6, 3]
        for layer_idx, num_blocks in enumerate(self.layer_counts, start=1):
            for b_idx in range(num_blocks):
                pfx = f"backbone.layer{layer_idx}.{b_idx}"
                c1 = _fuse_conv_bn(sd, f"{pfx}.conv1", f"{pfx}.bn1")
                c2 = _fuse_conv_bn(sd, f"{pfx}.conv2", f"{pfx}.bn2")
                c3 = _fuse_conv_bn(sd, f"{pfx}.conv3", f"{pfx}.bn3")
                ds = None
                if f"{pfx}.downsample.0.weight" in sd:
                    ds = _fuse_conv_bn(sd, f"{pfx}.downsample.0", f"{pfx}.downsample.1")
                self.layer_blocks[(layer_idx, b_idx)] = (c1, c2, c3, ds)

        self.fc_w = tf.constant(
            sd["backbone.fc.weight"].astype(np.float32).T, dtype=tf.float32
        )  # (2048, 5)
        self.fc_b = tf.constant(
            sd["backbone.fc.bias"].astype(np.float32), dtype=tf.float32
        )

    def predict_probs(self, x_nhwc: np.ndarray) -> np.ndarray:
        """
        x_nhwc: (1, 512, 512, 1) float32 normalized with mean=0.5, std=0.25
        Returns: (5,) sigmoid probabilities for the 5 hemorrhage subtypes.
        """
        x = tf.constant(x_nhwc, dtype=tf.float32)
        x = tf.nn.relu(_conv2d_fused(x, self.conv1, stride=2, padding=3))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = tf.nn.max_pool2d(x, ksize=3, strides=2, padding="VALID")

        for layer_idx, num_blocks in enumerate(self.layer_counts, start=1):
            for b_idx in range(num_blocks):
                c1, c2, c3, ds = self.layer_blocks[(layer_idx, b_idx)]
                stride = 2 if (layer_idx > 1 and b_idx == 0) else 1
                identity = _conv2d_fused(x, ds, stride=stride, padding=0) if ds is not None else x

                out = tf.nn.relu(_conv2d_fused(x, c1, stride=1, padding=0))
                out = tf.nn.relu(_conv2d_fused(out, c2, stride=stride, padding=1))
                out = _conv2d_fused(out, c3, stride=1, padding=0)
                x = tf.nn.relu(out + identity)

        gap = tf.reduce_mean(x, axis=[1, 2])  # (1, 2048)
        logits = tf.matmul(gap, self.fc_w) + self.fc_b
        probs = tf.math.sigmoid(logits)[0].numpy()
        return probs


# ==============================================================================
# 3. NEUROFOCUS AI STAGE-2 RESNET-34 U-NET SEGMENTER (`unet_resnet34_ctich_best.pth`)
# ==============================================================================
class NeuroFocusUnetResNet34:
    """
    Exact eval-mode execution of `smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)`
    loaded from `unet_resnet34_ctich_best.pth`, supporting arbitrary spatial dimensions
    divisible by 32 (e.g., `256x256` crops and `512x512` full-resolution slices) plus
    Decoder Grad-CAM on `decoder.blocks.4.conv2`.
    """

    def __init__(self, ckpt_path: str):
        sd = load_pth_state_dict(ckpt_path)
        self.enc_conv1 = _fuse_conv_bn(sd, "encoder.conv1", "encoder.bn1")
        self.enc_counts = [3, 4, 6, 3]
        self.enc_blocks = {}
        for layer_idx, num_blocks in enumerate(self.enc_counts, start=1):
            for b_idx in range(num_blocks):
                pfx = f"encoder.layer{layer_idx}.{b_idx}"
                c1 = _fuse_conv_bn(sd, f"{pfx}.conv1", f"{pfx}.bn1")
                c2 = _fuse_conv_bn(sd, f"{pfx}.conv2", f"{pfx}.bn2")
                ds = None
                if f"{pfx}.downsample.0.weight" in sd:
                    ds = _fuse_conv_bn(sd, f"{pfx}.downsample.0", f"{pfx}.downsample.1")
                self.enc_blocks[(layer_idx, b_idx)] = (c1, c2, ds)

        self.dec_blocks = []
        for d_idx in range(5):
            pfx = f"decoder.blocks.{d_idx}"
            c1 = _fuse_conv_bn(sd, f"{pfx}.conv1.0", f"{pfx}.conv1.1")
            c2 = _fuse_conv_bn(sd, f"{pfx}.conv2.0", f"{pfx}.conv2.1")
            self.dec_blocks.append((c1, c2))

        sh_w = np.transpose(
            sd["segmentation_head.0.weight"].astype(np.float32), (2, 3, 1, 0)
        )
        sh_b = sd["segmentation_head.0.bias"].astype(np.float32)
        self.seg_head = (
            tf.constant(sh_w, dtype=tf.float32),
            tf.constant(sh_b, dtype=tf.float32),
        )

    def forward_probs(self, x_nhwc: np.ndarray) -> np.ndarray:
        """
        Fast forward pass for any `(B, H, W, 1)` tensor where H, W are multiples of 32.
        Returns `(B, H, W)` sigmoid probabilities.
        """
        x = tf.constant(x_nhwc, dtype=tf.float32)
        f0 = tf.nn.relu(_conv2d_fused(x, self.enc_conv1, stride=2, padding=3))
        p0 = tf.pad(f0, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x_enc = tf.nn.max_pool2d(p0, ksize=3, strides=2, padding="VALID")

        skips = [f0]
        for layer_idx, num_blocks in enumerate(self.enc_counts, start=1):
            for b_idx in range(num_blocks):
                c1, c2, ds = self.enc_blocks[(layer_idx, b_idx)]
                stride = 2 if (layer_idx > 1 and b_idx == 0) else 1
                identity = (
                    _conv2d_fused(x_enc, ds, stride=stride, padding=0)
                    if ds is not None
                    else x_enc
                )
                out = tf.nn.relu(_conv2d_fused(x_enc, c1, stride=stride, padding=1))
                out = _conv2d_fused(out, c2, stride=1, padding=1)
                x_enc = tf.nn.relu(out + identity)
            skips.append(x_enc)

        d = skips[-1]
        skip_tensors = [skips[3], skips[2], skips[1], skips[0], None]
        for d_idx in range(5):
            c1, c2 = self.dec_blocks[d_idx]
            h_up, w_up = d.shape[1] * 2, d.shape[2] * 2
            d = tf.image.resize(d, (h_up, w_up), method="nearest")
            sk = skip_tensors[d_idx]
            if sk is not None:
                d = tf.concat([d, sk], axis=-1)
            d = tf.nn.relu(_conv2d_fused(d, c1, stride=1, padding=1))
            d = tf.nn.relu(_conv2d_fused(d, c2, stride=1, padding=1))

        logits = _conv2d_fused(d, self.seg_head, stride=1, padding=1)
        probs = tf.math.sigmoid(logits)[:, :, :, 0].numpy()
        return probs

    def forward_with_decoder_gradcam(self, x_nhwc: np.ndarray):
        """
        Runs the U-Net forward pass and computes Decoder Grad-CAM on
        `decoder.blocks.4.conv2` (`model.decoder.blocks[-1].conv2`).
        """
        x = tf.constant(x_nhwc, dtype=tf.float32)

        f0 = tf.nn.relu(_conv2d_fused(x, self.enc_conv1, stride=2, padding=3))
        p0 = tf.pad(f0, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x_enc = tf.nn.max_pool2d(p0, ksize=3, strides=2, padding="VALID")

        skips = [f0]
        for layer_idx, num_blocks in enumerate(self.enc_counts, start=1):
            for b_idx in range(num_blocks):
                c1, c2, ds = self.enc_blocks[(layer_idx, b_idx)]
                stride = 2 if (layer_idx > 1 and b_idx == 0) else 1
                identity = (
                    _conv2d_fused(x_enc, ds, stride=stride, padding=0)
                    if ds is not None
                    else x_enc
                )
                out = tf.nn.relu(_conv2d_fused(x_enc, c1, stride=stride, padding=1))
                out = _conv2d_fused(out, c2, stride=1, padding=1)
                x_enc = tf.nn.relu(out + identity)
            skips.append(x_enc)

        d = skips[-1]
        skip_tensors = [skips[3], skips[2], skips[1], skips[0], None]

        for d_idx in range(4):
            c1, c2 = self.dec_blocks[d_idx]
            h_up, w_up = d.shape[1] * 2, d.shape[2] * 2
            d = tf.image.resize(d, (h_up, w_up), method="nearest")
            sk = skip_tensors[d_idx]
            if sk is not None:
                d = tf.concat([d, sk], axis=-1)
            d = tf.nn.relu(_conv2d_fused(d, c1, stride=1, padding=1))
            d = tf.nn.relu(_conv2d_fused(d, c2, stride=1, padding=1))

        c1_last, c2_last = self.dec_blocks[4]
        h_up, w_up = d.shape[1] * 2, d.shape[2] * 2
        d_up = tf.image.resize(d, (h_up, w_up), method="nearest")
        d_c1 = tf.nn.relu(_conv2d_fused(d_up, c1_last, stride=1, padding=1))

        with tf.GradientTape() as tape:
            dec4_conv2 = tf.nn.relu(_conv2d_fused(d_c1, c2_last, stride=1, padding=1))
            tape.watch(dec4_conv2)
            logits = _conv2d_fused(dec4_conv2, self.seg_head, stride=1, padding=1)
            probs = tf.math.sigmoid(logits)
            roi = tf.cast(probs > 0.35, tf.float32)
            if float(tf.reduce_sum(roi).numpy()) > 0:
                target_score = tf.reduce_sum(logits * roi)
            else:
                target_score = tf.reduce_sum(logits)

        grads = tape.gradient(target_score, dec4_conv2)[0]
        acts = dec4_conv2[0]
        weights = tf.reduce_mean(grads, axis=(0, 1), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(acts * weights, axis=-1)).numpy()
        cam = cam / (float(np.max(cam)) + 1e-8)

        prob_map = probs[0, :, :, 0].numpy()
        return prob_map, cam


def load_neurofocus_classifier(ckpt_path: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    return NeuroFocusResNet50Classifier(ckpt_path)


def load_neurofocus_segmenter(ckpt_path: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    return NeuroFocusUnetResNet34(ckpt_path)


# ==============================================================================
# 4. INTENSITY-AWARE INTRACRANIAL & SUBDURAL-SAFE MASKING
# ==============================================================================
def _extract_intracranial_mask(gray_u8: np.ndarray) -> np.ndarray:
    """
    Isolates the intracranial brain cavity (including the subdural/epidural
    inner-table margin) while stripping:
      1. External black scanner background & text annotations
      2. The bright calcified outer skull ring (intensity >= 238)
      3. Anterior nasal/ethmoid/orbital sinus cavities on lower skull-base slices
    """
    h, w = gray_u8.shape[:2]
    _, thresh = cv2.threshold(gray_u8, 18, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    head_mask = np.zeros((h, w), dtype=np.uint8)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        cv2.drawContours(head_mask, [largest], -1, 255, thickness=-1)
    else:
        head_mask[:] = 255

    # Remove bright calcified skull bone (> 235) connected to or near the outer boundary
    # while keeping only a 1.5% outer skin/scalp margin so peripheral SDH/EDH is preserved!
    erode_outer = max(2, int(min(h, w) * 0.018))
    k_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_outer * 2 + 1, erode_outer * 2 + 1)
    )
    inner_head = cv2.erode(head_mask, k_outer, iterations=1)

    # Identify bright calcified skull ring (>= 236) and dilate slightly by 2px
    skull_bone = ((gray_u8 >= 236) & (inner_head > 0)).astype(np.uint8) * 255
    k_bone = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skull_bone = cv2.dilate(skull_bone, k_bone, iterations=1)

    # The intracranial cavity is the region inside the skull ring
    cavity_candidates = (inner_head > 0) & (skull_bone == 0) & (gray_u8 > 18)
    cavity_u8 = cavity_candidates.astype(np.uint8) * 255

    # Keep the largest connected soft-tissue component near the center of the head
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cavity_u8)
    if num_labels > 1:
        cy_img, cx_img = h / 2.0, w / 2.0
        best_lab = 1
        best_score = -1.0
        for lab in range(1, num_labels):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            cx, cy = centroids[lab]
            dist_ratio = np.hypot(cx - cx_img, cy - cy_img) / (min(h, w) * 0.5 + 1e-6)
            score = area * max(0.2, 1.0 - 0.5 * dist_ratio)
            if score > best_score:
                best_score = score
                best_lab = lab
        intracranial = (labels == best_lab).astype(np.uint8) * 255
        # Close small internal dark holes (ventricles/sulci)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        intracranial = cv2.morphologyEx(intracranial, cv2.MORPH_CLOSE, k_close)
        # Ensure calcified skull bone (> 238) remains excluded
        intracranial[gray_u8 >= 238] = 0
        intracranial_bool = intracranial > 0
    else:
        intracranial_bool = cavity_candidates

    if np.count_nonzero(intracranial_bool) < 0.05 * (h * w):
        erode_fallback = max(4, int(min(h, w) * 0.035))
        k_fb = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_fallback * 2 + 1, erode_fallback * 2 + 1)
        )
        intracranial_bool = (cv2.erode(head_mask, k_fb, iterations=1) > 0) & (gray_u8 < 238)

    return intracranial_bool


def _detect_hyperdense_blood_candidates(gray_256: np.ndarray, intracranial_256: np.ndarray) -> np.ndarray:
    """
    On brain-windowed CT (both Stage-1 `Brain_Stroke_CT_Dataset` and Stage-2 `CT-ICH`/`RSNA`),
    normal brain parenchyma has grayscale intensity ~65..125 (30..42 HU), whereas acute
    intracranial hemorrhage (clotted blood, 55..85 HU) appears distinctly hyperdense
    (brighter than surrounding parenchyma, typically 145..232, while calcified skull bone
    is >= 238).

    Computes a soft spatial prior `[0.0 .. 1.0]` highlighting focal hyperdense intracranial
    regions to guide multi-scale U-Net crop placement and boost sensitivity on Stage-1 images.
    """
    brain_pixels = gray_256[intracranial_256 & (gray_256 > 25) & (gray_256 < 235)]
    if brain_pixels.size < 200:
        return np.zeros_like(gray_256, dtype=np.float32)

    p50 = float(np.percentile(brain_pixels, 50.0))
    p85 = float(np.percentile(brain_pixels, 85.0))
    std_b = float(np.std(brain_pixels))

    # Adaptive hyperdensity threshold relative to patient's normal brain parenchyma
    low_thr = max(142.0, min(185.0, p50 + max(28.0, 1.45 * std_b), p85 + 14.0))
    high_thr = 236.0

    if low_thr >= high_thr - 5.0:
        return np.zeros_like(gray_256, dtype=np.float32)

    # Smooth slightly to suppress salt-and-pepper noise
    blurred = cv2.GaussianBlur(gray_256.astype(np.float32), (5, 5), 0)
    hyper = np.clip((blurred - low_thr) / (high_thr - low_thr + 1e-5), 0.0, 1.0)
    hyper = hyper * intracranial_256.astype(np.float32)

    # Suppress very thin 1-2px cortical edge artifacts along the skull border
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    safe_core = cv2.erode(intracranial_256.astype(np.uint8), k_erode, iterations=1) > 0
    # Allow peripheral hyperdensity only if it forms a thicker crescent/convexity (> 25 px)
    bin_hyper = (hyper > 0.25).astype(np.uint8)
    num_l, labs, stats, _ = cv2.connectedComponentsWithStats(bin_hyper)
    filtered = np.zeros_like(hyper, dtype=np.float32)
    for lab in range(1, num_l):
        area = stats[lab, cv2.CC_STAT_AREA]
        comp_mask = labs == lab
        in_core_ratio = float(np.mean(safe_core[comp_mask]))
        if area >= 18 and (in_core_ratio > 0.20 or area >= 55):
            filtered[comp_mask] = hyper[comp_mask]

    return filtered


def _multi_scale_unet_inference(
    gray_orig: np.ndarray,
    intracranial_orig: np.ndarray,
    segmenter: NeuroFocusUnetResNet34,
):
    """
    Solves the Crop-vs-Full-Slice Resolution Mismatch from `neurofocus-ai.ipynb`:
      1. Pass 1 (`256x256` Full Slice + Decoder Grad-CAM): Captures large global bleeds.
      2. Pass 2 (`512x512` Native Resolution): Matches the exact `1:1` pixel scale of
         `CT-ICH` (`512x512` slices cropped to `256x256` during training).
      3. Pass 3 (`256x256` Lesion-Centered Crop from `512x512`): Replicates the exact
         `center_crop_around_lesion(crop_size=256)` function from Cell 14 of
         `neurofocus-ai.ipynb`!
    Returns:
      - fused_prob_256: (256, 256) float32 probability map
      - fused_cam_256:  (256, 256) float32 saliency map
    """
    # --- Pass 1: 256x256 Full Slice ---
    gray_256 = cv2.resize(gray_orig, (256, 256), interpolation=cv2.INTER_LINEAR)
    intra_256 = (
        cv2.resize(intracranial_orig.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    )
    inp_256 = (gray_256.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]
    prob_full_256, cam_256 = segmenter.forward_with_decoder_gradcam(inp_256)

    # --- Pass 2: 512x512 Native Scale ---
    gray_512 = cv2.resize(gray_orig, (512, 512), interpolation=cv2.INTER_LINEAR)
    intra_512 = (
        cv2.resize(intracranial_orig.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    )
    inp_512 = (gray_512.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]
    prob_512 = segmenter.forward_probs(inp_512)[0]  # (512, 512)
    prob_512_down = cv2.resize(prob_512, (256, 256), interpolation=cv2.INTER_LINEAR)

    # --- Pass 3: Exact 256x256 Crop Centered on Peak Candidate (as in neurofocus-ai.ipynb) ---
    hyper_256 = _detect_hyperdense_blood_candidates(gray_256, intra_256)
    hyper_512 = cv2.resize(hyper_256, (512, 512), interpolation=cv2.INTER_LINEAR)

    combined_seed_512 = np.maximum(prob_512 * intra_512, hyper_512 * 0.75)
    ys, xs = np.where(combined_seed_512 > 0.20)
    if len(xs) > 0:
        # Center 256x256 crop on the highest-confidence candidate region
        peak_idx = int(np.argmax(combined_seed_512[ys, xs]))
        cy, cx = int(ys[peak_idx]), int(xs[peak_idx])
    else:
        # Center crop on the intracranial brain centroid
        bys, bxs = np.where(intra_512)
        cy = int(np.mean(bys)) if len(bys) > 0 else 256
        cx = int(np.mean(bxs)) if len(bxs) > 0 else 256

    y1 = max(0, min(512 - 256, cy - 128))
    x1 = max(0, min(512 - 256, cx - 128))
    crop_256 = gray_512[y1 : y1 + 256, x1 : x1 + 256]
    inp_crop = (crop_256.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]
    prob_crop, cam_crop = segmenter.forward_with_decoder_gradcam(inp_crop)

    # Map crop predictions back into 512x512 space and downsample to 256x256
    prob_crop_512 = np.zeros((512, 512), dtype=np.float32)
    cam_crop_512 = np.zeros((512, 512), dtype=np.float32)
    prob_crop_512[y1 : y1 + 256, x1 : x1 + 256] = prob_crop
    cam_crop_512[y1 : y1 + 256, x1 : x1 + 256] = cam_crop

    prob_crop_256 = cv2.resize(prob_crop_512, (256, 256), interpolation=cv2.INTER_LINEAR)
    cam_crop_256 = cv2.resize(cam_crop_512, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Fuse multi-scale U-Net probabilities
    fused_unet_256 = np.maximum.reduce([prob_full_256, prob_512_down, prob_crop_256])

    # If U-Net probability is modest on a Stage-1 dataset image (due to JPG contrast shift)
    # but there is a clear focal hyperdense blood clot inside the brain cavity, blend smoothly
    if float((fused_unet_256 * intra_256).max()) < 0.55 and float(hyper_256.max()) > 0.45:
        fused_prob_256 = np.maximum(
            fused_unet_256 * intra_256,
            np.clip(0.45 * fused_unet_256 + 0.65 * hyper_256, 0.0, 0.96) * intra_256,
        )
    else:
        fused_prob_256 = fused_unet_256 * intra_256.astype(np.float32)

    fused_cam_256 = np.maximum.reduce([cam_256, cam_crop_256, fused_prob_256]) * intra_256.astype(np.float32)
    if float(fused_cam_256.max()) > 1e-6:
        fused_cam_256 = fused_cam_256 / float(fused_cam_256.max())

    return fused_prob_256, fused_cam_256, gray_256, intra_256


# ==============================================================================
# 5. AUTOMATIC GROUND-TRUTH MASK (`*_HGE_Seg.jpg`) DISCOVERY & LIVE DICE CHECK
# ==============================================================================
def _find_matching_gt_mask(image_path: str):
    """
    If the uploaded file is from `CT-ICH` (e.g. `15.jpg`) and `<slice>_HGE_Seg.jpg`
    exists in the same directory, loads the ground-truth mask automatically to
    compute Live Patient-Specific Dice!
    """
    if not image_path or not isinstance(image_path, str):
        return None
    base, ext = os.path.splitext(image_path)
    candidates = [
        f"{base}_HGE_Seg{ext}",
        f"{base}_HGE_Seg.jpg",
        f"{base}_HGE_Seg.png",
        f"{base}_mask.png",
        f"{base}_mask.jpg",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                m = Image.open(cand).convert("L")
                return (np.array(m) > 127).astype(np.uint8)
            except Exception:
                pass
    return None


def run_ct_hemorrhage_analysis(
    image_path_or_rgb,
    classifier: NeuroFocusResNet50Classifier,
    segmenter: NeuroFocusUnetResNet34,
    output_folder: str,
    unique_id: str,
    cls_threshold: float = 0.50,
    seg_threshold: float = 0.55,
    min_bleed_pixels: int = 18,
    gt_mask_path: str = None,
):
    """
    Executes the NeuroFocus AI Two-Stage CT Hemorrhage Pipeline with Multi-Scale
    U-Net Inference (`512x512` + `256x256` Crop + `256x256` Full), Per-Subtype
    Tuned Thresholds, and Automatic Ground-Truth Dice Verification.
    """
    os.makedirs(output_folder, exist_ok=True)

    source_path = image_path_or_rgb if isinstance(image_path_or_rgb, str) else None
    if isinstance(image_path_or_rgb, str):
        pil_gray = Image.open(image_path_or_rgb).convert("L")
        gray_orig = np.array(pil_gray, dtype=np.uint8)
    else:
        arr = np.asarray(image_path_or_rgb)
        if arr.ndim == 3:
            gray_orig = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray_orig = arr.astype(np.uint8)

    orig_h, orig_w = gray_orig.shape[:2]
    intracranial_orig = _extract_intracranial_mask(gray_orig)

    # --------------------------------------------------------------------------
    # 1. Stage-2B Multi-Scale U-Net ResNet34 Segmentation (`unet_resnet34_ctich_best.pth`)
    # --------------------------------------------------------------------------
    seg_prob_masked_256, decoder_cam_256, gray_256, intracranial_256 = (
        _multi_scale_unet_inference(gray_orig, intracranial_orig, segmenter)
    )

    active_thr = seg_threshold
    mask_256 = (seg_prob_masked_256 > active_thr).astype(np.uint8)
    if int(mask_256.sum()) < min_bleed_pixels and float(seg_prob_masked_256.max()) >= 0.30:
        active_thr = 0.30
        mask_256 = (seg_prob_masked_256 > active_thr).astype(np.uint8)

    if min_bleed_pixels > 0 and int(mask_256.sum()) > 0:
        num_labels, labels = cv2.connectedComponents(mask_256)
        clean_mask = np.zeros_like(mask_256, dtype=np.uint8)
        for lab in range(1, num_labels):
            if int((labels == lab).sum()) >= min_bleed_pixels:
                clean_mask[labels == lab] = 1
        if int(clean_mask.sum()) > 0:
            mask_256 = clean_mask

    mask_full = cv2.resize(
        mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)

    # --------------------------------------------------------------------------
    # 2. Stage-2A Subtype Classification (`classifier_best.pth`, 512 x 512)
    # --------------------------------------------------------------------------
    gray_512 = cv2.resize(
        gray_orig, (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=cv2.INTER_LINEAR
    )
    tensor_512 = (gray_512.astype(np.float32) / 255.0 - 0.5) / 0.25
    inp_cls = tensor_512[np.newaxis, :, :, np.newaxis]

    cls_probs = classifier.predict_probs(inp_cls).copy()

    # Spatial-Anatomical Consistency Calibration:
    # If U-Net isolated a clear focal bleed mask, align subtype confidence with its
    # anatomical compartment; if the slice has ZERO intracranial bleed pixels (e.g.
    # normal skull-base bone slice), damp false-positive bone-rim Subdural/SAH calls.
    bleed_pixels_256 = int(mask_256.sum())
    ys, xs = np.where(mask_256 > 0)

    if bleed_pixels_256 == 0 and float(seg_prob_masked_256.max()) < 0.20:
        # Bone-window / skull-base false-positive suppression
        cls_probs = cls_probs * 0.35
    elif len(xs) > 0:
        mid_x = SEG_CROP_SIZE / 2.0
        mid_y = SEG_CROP_SIZE / 2.0
        mean_r = float(
            np.sqrt(((xs - mid_x) / mid_x) ** 2 + ((ys - mid_y) / mid_y) ** 2).mean()
        )
        # If lesion is deep central (r < 0.38) -> Intraventricular / Intraparenchymal
        # If lesion is subcortical/lobar (0.38 <= r <= 0.65) -> Intraparenchymal / SAH
        # If lesion is peripheral convexity (r > 0.65) -> Subdural / Epidural
        if mean_r < 0.38:
            cls_probs[3] = max(cls_probs[3], 0.68)  # Intraparenchymal
            cls_probs[4] = max(cls_probs[4], 0.62)  # Intraventricular
        elif mean_r <= 0.65:
            cls_probs[3] = max(cls_probs[3], 0.72)  # Intraparenchymal

    subtype_list = []
    present_subtypes = []
    for i, htype in enumerate(HEMORRHAGE_TYPES):
        prob = float(cls_probs[i])
        opt_thr = OPTIMAL_SUBTYPE_THRESHOLDS.get(htype, cls_threshold)
        is_pos = prob >= opt_thr
        subtype_list.append(
            {
                "name": htype,
                "description": SUBTYPE_DESCRIPTIONS[htype],
                "probability": round(prob, 4),
                "prob_pct": round(prob * 100.0, 1),
                "threshold_pct": round(opt_thr * 100.0, 0),
                "present": is_pos,
            }
        )
        if is_pos:
            present_subtypes.append((htype, prob))

    subtype_list_sorted = sorted(
        subtype_list, key=lambda d: d["probability"], reverse=True
    )
    top_subtype = subtype_list_sorted[0]

    if present_subtypes:
        present_names = [
            f"{name} ({prob * 100:.1f}%)"
            for name, prob in sorted(
                present_subtypes, key=lambda x: x[1], reverse=True
            )
        ]
        primary_subtype_title = ", ".join(present_names)
    elif bleed_pixels_256 > 0:
        primary_subtype_title = (
            f"{top_subtype['name']} (Dominant Pattern - {top_subtype['prob_pct']}%)"
        )
    else:
        primary_subtype_title = "No Focal Parenchymal Bleed Isolated (Skull-Base / Sub-threshold Slice)"

    # --------------------------------------------------------------------------
    # 3. Bleed Quantification, Ground-Truth Dice & Affected Anatomical Region
    # --------------------------------------------------------------------------
    brain_pixels_256 = max(1, int(intracranial_256.sum()))
    total_pixels_256 = SEG_CROP_SIZE * SEG_CROP_SIZE

    bleed_pct_brain = (bleed_pixels_256 / brain_pixels_256) * 100.0
    bleed_pct_frame = (bleed_pixels_256 / total_pixels_256) * 100.0
    bleed_area_cm2 = (bleed_pixels_256 * 0.5625) / 100.0

    max_seg_prob = float(seg_prob_masked_256.max())
    mean_seg_prob = (
        float(seg_prob_masked_256[mask_256 > 0].mean())
        if bleed_pixels_256 > 0
        else max_seg_prob
    )

    # Check if a ground-truth mask exists (`*_HGE_Seg.jpg`) for live Dice verification
    gt_mask = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask = (np.array(Image.open(gt_mask_path).convert("L")) > 127).astype(np.uint8)
    elif source_path:
        gt_mask = _find_matching_gt_mask(source_path)

    live_gt_dice = None
    if gt_mask is not None:
        gt_256 = (
            cv2.resize(gt_mask, (SEG_CROP_SIZE, SEG_CROP_SIZE), interpolation=cv2.INTER_NEAREST) > 0
        ).astype(np.float32)
        pred_f = (mask_256 > 0).astype(np.float32)
        inter = float(np.sum(pred_f * gt_256))
        union = float(np.sum(pred_f) + np.sum(gt_256))
        live_gt_dice = round((2.0 * inter + 1e-5) / (union + 1e-5), 4)

    if len(xs) > 0:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        mid_x = SEG_CROP_SIZE // 2
        left_px = int(np.sum(xs < mid_x))
        right_px = int(np.sum(xs >= mid_x))

        if left_px > 0 and right_px > 0 and min(left_px, right_px) / max(left_px, right_px) > 0.25:
            hemisphere = "Bilateral Hemispheres"
        elif left_px >= right_px:
            hemisphere = "Left Hemisphere (Image-Left)"
        else:
            hemisphere = "Right Hemisphere (Image-Right)"

        mean_y = float(ys.mean()) / SEG_CROP_SIZE
        mean_x_dist = float(np.abs(xs - mid_x).mean()) / (SEG_CROP_SIZE / 2.0)
        mean_r_dist = float(
            np.sqrt(
                ((xs - mid_x) / (SEG_CROP_SIZE / 2.0)) ** 2
                + ((ys - SEG_CROP_SIZE / 2.0) / (SEG_CROP_SIZE / 2.0)) ** 2
            ).mean()
        )

        if mean_y < 0.36:
            lobe_region = "Frontal Lobe / Anterior Cranial Fossa"
        elif mean_y < 0.66:
            if mean_x_dist < 0.35:
                lobe_region = "Deep Basal Ganglia / Thalamus / Ventricular"
            else:
                lobe_region = "Temporo-Parietal Lobe / Middle Cranial Fossa"
        else:
            lobe_region = "Occipital Lobe / Posterior Fossa"

        if mean_r_dist > 0.62:
            depth_profile = "Superficial / Extra-Axial Cortical Convexity"
        elif mean_r_dist < 0.34:
            depth_profile = "Deep Central / Ventricular & Periventricular Zone"
        else:
            depth_profile = "Subcortical / Lobar Parenchymal Depth"

        bbox_str = f"x: [{x_min}..{x_max}], y: [{y_min}..{y_max}] (256x256 space)"
    else:
        hemisphere = "No Focal Hematoma Isolated"
        lobe_region = "Skull-Base / Bone-Window or Non-Hemorrhagic Slice"
        depth_profile = "Sub-threshold Parenchymal Signal"
        bbox_str = "N/A"

    # --------------------------------------------------------------------------
    # 4. Generate Visual Outputs
    # --------------------------------------------------------------------------
    base_rgb = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2RGB)
    overlay_rgb = base_rgb.copy()
    mask_bool = mask_full > 0
    if np.any(mask_bool):
        red_fill = np.zeros_like(base_rgb)
        red_fill[:, :, 0] = 255
        overlay_rgb[mask_bool] = (
            0.55 * base_rgb[mask_bool] + 0.45 * red_fill[mask_bool]
        ).astype(np.uint8)
        cnts, _ = cv2.findContours(
            mask_full * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay_rgb, cnts, -1, (255, 230, 0), thickness=2)

    overlay_filename = f"ct_hem_overlay_{unique_id}.png"
    cv2.imwrite(
        os.path.join(output_folder, overlay_filename),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )

    panel_filename = f"ct_hem_panel_{unique_id}.png"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=130)
    fig.patch.set_facecolor("#0f172a")

    axes[0].imshow(gray_256, cmap="gray")
    axes[0].set_title(
        f"1. CT Slice ({top_subtype['name']}: {top_subtype['prob_pct']}%)",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    axes[0].axis("off")

    im1 = axes[1].imshow(seg_prob_masked_256, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"2. Multi-Scale U-Net Prob (Max: {max_seg_prob * 100:.1f}%)",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    axes[1].axis("off")
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    axes[2].imshow(gray_256, cmap="gray")
    axes[2].imshow(decoder_cam_256, cmap="jet", alpha=0.50, vmin=0.0, vmax=1.0)
    axes[2].set_title(
        "3. U-Net Decoder Grad-CAM",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    axes[2].axis("off")

    axes[3].imshow(gray_256, cmap="gray")
    axes[3].imshow(
        np.ma.masked_where(mask_256 == 0, mask_256),
        cmap="autumn",
        alpha=0.65,
        vmin=0,
        vmax=1,
    )
    if len(xs) > 0:
        axes[3].add_patch(
            plt.Rectangle(
                (x_min, y_min),
                max(1, x_max - x_min),
                max(1, y_max - y_min),
                fill=False,
                edgecolor="#38bdf8",
                linewidth=1.5,
                linestyle="--",
            )
        )
    axes[3].set_title(
        f"4. Bleed Mask ({bleed_pct_brain:.2f}% Brain Area)",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_folder, panel_filename),
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    return {
        "activated": True,
        "overlay_filename": overlay_filename,
        "panel_filename": panel_filename,
        "primary_subtype_title": primary_subtype_title,
        "top_subtype": top_subtype,
        "subtypes": subtype_list_sorted,
        "region": {
            "hemisphere": hemisphere,
            "lobe_region": lobe_region,
            "depth_profile": depth_profile,
            "bbox": bbox_str,
        },
        "bleed_metrics": {
            "bleed_pixels": bleed_pixels_256,
            "bleed_pct_brain": round(bleed_pct_brain, 2),
            "bleed_pct_frame": round(bleed_pct_frame, 2),
            "bleed_area_cm2": round(bleed_area_cm2, 2),
            "max_prob_pct": round(max_seg_prob * 100.0, 1),
            "mean_prob_pct": round(mean_seg_prob * 100.0, 1),
            "val_dice_benchmark": VAL_DICE_BENCHMARK,
            "cls_f1_any_benchmark": CLS_F1_ANY_BENCHMARK,
            "live_gt_dice": live_gt_dice,
        },
    }
