import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image

import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from rag_client import query_clinical_guidelines, ask_clinical_qa


# ----------------------------
# App + DB setup
# ----------------------------
app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(os.path.join(os.path.dirname(__file__), "instance"), exist_ok=True)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"


# ----------------------------
# IMPORTANT: edit to match your ML model columns
# ----------------------------
NUM_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

# IMPORTANT: your folder order is correct (alphabetical)
MRI_CLASS_NAMES = ["Haemorrhagic", "Ischemic", "Normal"]
CT_CLASS_NAMES = ["Hemorrhage", "Ischemia", "Normal"]

MRI_IMG_SIZE = (224, 224)
CT_IMG_SIZE = (224, 224)


# ----------------------------
# Database Models
# ----------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    pred_type = db.Column(db.String(30), nullable=False)  # "numerical" | "image_mri" | "image_ct" | "image"(old)

    # numerical
    input_json = db.Column(db.Text, nullable=True)
    proba = db.Column(db.Float, nullable=True)
    label = db.Column(db.String(50), nullable=True)

    # image
    image_filename = db.Column(db.String(255), nullable=True)
    gradcam_filename = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="predictions")


with app.app_context():
    db.create_all()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ----------------------------
# Utilities
# ----------------------------
def now_year():
    return datetime.now().year


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in app.config["ALLOWED_EXTENSIONS"]


_ml_model_cache = None
_mri_model_cache = None
_ct_model_cache = None
_lesion_model_cache = None
_ct_subtype_model_cache = None
_ct_seg_model_cache = None


def load_ml_model():
    global _ml_model_cache
    if _ml_model_cache is not None:
        return _ml_model_cache
    path = app.config["MODEL_ML_PATH"]
    if not os.path.exists(path):
        return None
    _ml_model_cache = joblib.load(path)
    return _ml_model_cache


def load_mri_model():
    global _mri_model_cache
    if _mri_model_cache is not None: return _mri_model_cache
    path = app.config["MODEL_MRI_PATH"]
    if not os.path.exists(path): return None
    _mri_model_cache = tf.keras.models.load_model(path, compile=False)
    return _mri_model_cache

def load_ct_model():
    global _ct_model_cache
    if _ct_model_cache is not None: return _ct_model_cache
    path = app.config["MODEL_CT_PATH"]
    if not os.path.exists(path): return None
    _ct_model_cache = tf.keras.models.load_model(path, compile=False)
    return _ct_model_cache


def load_lesion_model():
    """Load Stage-2 Ischemic Lesion Segmentation U-Net++ model (best_stroke_model.keras)."""
    global _lesion_model_cache
    if _lesion_model_cache is not None:
        return _lesion_model_cache
    path = app.config.get("MODEL_LESION_PATH")
    if not path or not os.path.exists(path):
        return None
    _lesion_model_cache = tf.keras.models.load_model(path, compile=False)
    return _lesion_model_cache


def load_ct_neurofocus_models():
    """Load NeuroFocus AI Stage-2 CT Hemorrhage Subtype Classifier & ResNet-34 U-Net Segmenter."""
    global _ct_subtype_model_cache, _ct_seg_model_cache
    if _ct_subtype_model_cache is not None and _ct_seg_model_cache is not None:
        return _ct_subtype_model_cache, _ct_seg_model_cache
    from ct_hemorrhage_analyzer import (
        load_neurofocus_classifier,
        load_neurofocus_segmenter,
    )
    cls_path = app.config.get("MODEL_CT_SUBTYPE_PATH")
    seg_path = app.config.get("MODEL_CT_SEG_PATH")
    if _ct_subtype_model_cache is None:
        _ct_subtype_model_cache = load_neurofocus_classifier(cls_path)
    if _ct_seg_model_cache is None:
        _ct_seg_model_cache = load_neurofocus_segmenter(seg_path)
    return _ct_subtype_model_cache, _ct_seg_model_cache


# ----------------------------
# IMAGE PREPROCESS (FIX: avoid double-normalization)
# ----------------------------
def _model_has_rescaling(model):
    """Detect Rescaling(1./255) layer even inside nested models."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            return True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Rescaling):
                    return True
    return False


def prepare_image_input(model, img_rgb,img_size):
    """
    If model already has Rescaling layer (common in your training code),
    do NOT divide by 255 again.
    """
    img_resized = cv2.resize(img_rgb, img_size).astype(np.float32)

    if _model_has_rescaling(model):
        x = img_resized          # keep 0..255
    else:
        x = img_resized / 255.0  # scale to 0..1

    return np.expand_dims(x, axis=0)


# -------- Grad-CAM helpers (FIXED) --------
def _iter_layers_reverse(m):
    """Recursively yield layers in reverse order (supports nested models)."""
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.Model):
            yield from _iter_layers_reverse(layer)
        yield layer


def _find_multiscale_conv_layers(model):
    """
    Returns up to two Conv2D layers: the final Conv2D layer (7x7) and a higher-resolution
    intermediate Conv2D layer (14x14 or 28x28) so Grad-CAM does not collapse into a
    single coarse 7x7 center bullseye dot.
    """
    conv_layers = [
        layer for layer in _iter_layers_reverse(model)
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if not conv_layers:
        return []
    selected = [conv_layers[0]]
    for layer in conv_layers[1:]:
        try:
            shape = layer.output.shape
            if len(shape) == 4 and shape[1] is not None and int(shape[1]) in (14, 28):
                selected.append(layer)
                break
        except Exception:
            continue
    return selected


def make_gradcam_overlay(img_rgb, model, img_size, pred_index=None):
    """
    Multi-Scale Intracranial-Masked Grad-CAM:
      - Fuses 7x7 semantic activation with 14x14 / 28x28 spatial detail
      - Masks out external black background & outer calcified skull ring so
        activations highlight actual parenchymal lesions instead of a center dot.
    """
    target_layers = _find_multiscale_conv_layers(model)
    if not target_layers:
        raise ValueError("No Conv2D layer found for Grad-CAM (model must contain Conv2D layers).")

    model_inputs = model.inputs
    model_outputs = model.outputs
    pred_tensor = model_outputs[0] if isinstance(model_outputs, (list, tuple)) else model_outputs

    inp = prepare_image_input(model, img_rgb, img_size)
    inp_for_model = [inp] if isinstance(model_inputs, (list, tuple)) else inp

    base = cv2.resize(img_rgb, img_size)
    gray_base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)

    # Extract intracranial brain cavity mask to prevent corner/background artifacts
    _, head_thr = cv2.threshold(gray_base, 20, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(head_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_mask = np.zeros_like(gray_base, dtype=np.float32)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        cv2.drawContours(brain_mask, [largest], -1, 1.0, thickness=-1)
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        brain_mask = cv2.erode(brain_mask, k_erode, iterations=1)
        brain_mask[gray_base >= 238] = 0.0
    else:
        brain_mask[:] = 1.0

    fused_heatmap = np.zeros((img_size[1], img_size[0]), dtype=np.float32)

    for w_scale, conv_layer in zip([0.55, 0.45], target_layers):
        try:
            grad_model = tf.keras.models.Model(
                inputs=model_inputs,
                outputs=[conv_layer.output, pred_tensor]
            )
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(inp_for_model)
                preds = _unwrap_output(preds)
                if pred_index is None:
                    pred_index = int(tf.argmax(preds[0]))
                loss = preds[:, pred_index]

            grads = tape.gradient(loss, conv_out)
            if grads is None:
                continue
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            c_out = conv_out[0]
            hm = c_out @ pooled_grads[..., tf.newaxis]
            hm = tf.squeeze(hm)
            hm = tf.maximum(hm, 0).numpy()
            hm_resized = cv2.resize(hm, img_size, interpolation=cv2.INTER_CUBIC)
            hm_resized = hm_resized * brain_mask
            if float(np.max(hm_resized)) > 1e-8:
                hm_resized = hm_resized / float(np.max(hm_resized))
            fused_heatmap += w_scale * hm_resized
        except Exception:
            continue

    if float(np.max(fused_heatmap)) > 1e-8:
        fused_heatmap = fused_heatmap / float(np.max(fused_heatmap))

    heatmap_uint8 = np.uint8(255 * fused_heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.60,
        heatmap_color, 0.40, 0
    )
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay, pred_index


# ----------------------------
# Routes
# ----------------------------
@app.context_processor
def inject_globals():
    def static_file(filename: str) -> str:
        file_path = os.path.join(app.static_folder, filename)
        version = int(os.path.getmtime(file_path)) if os.path.exists(file_path) else None
        return url_for("static", filename=filename, v=version)

    return {"now_year": now_year(), "static_file": static_file}


@app.route("/")
def home():
    return render_template("home.html", title="Home")


@app.route("/about")
def about():
    return render_template("about.html", title="About")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not full_name or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("login"))

        user = User(
            full_name=full_name,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("auth_register.html", title="Register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        login_user(user)
        flash("Logged in successfully.", "success")
        return redirect(url_for("home"))

    return render_template("auth_login.html", title="Login")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "success")
    return redirect(url_for("home"))


@app.route("/predict/numerical", methods=["GET", "POST"])
@login_required
def predict_numerical():
    model = load_ml_model()
    if model is None:
        flash("Numerical model not found. Paste 'stroke_ml_model.pkl' into models/ folder.", "warning")

    result = None

    if request.method == "POST":
        if model is None:
            flash("Cannot predict: Numerical model missing.", "error")
            return redirect(url_for("predict_numerical"))

        payload = {}
        for f in NUM_FEATURES:
            payload[f] = request.form.get(f, None)

        # Cast numeric fields
        for f in ["age", "avg_glucose_level", "bmi"]:
            v = payload.get(f, None)
            if v is None or str(v).strip() == "":
                payload[f] = np.nan
            else:
                payload[f] = float(v)

        for f in ["hypertension", "heart_disease"]:
            payload[f] = int(payload.get(f, 0))

        X = pd.DataFrame([payload], columns=NUM_FEATURES)

        try:
            proba = float(model.predict_proba(X)[:, 1][0])
            pred = int(proba >= 0.5)
            label = "Stroke Risk" if pred == 1 else "No Stroke Risk"

            rec = PredictionHistory(
                user_id=current_user.id,
                pred_type="numerical",
                input_json=str(payload),
                proba=proba,
                label=label
            )
            db.session.add(rec)
            db.session.commit()

            result = {
                "proba": proba,
                "label": label,
                "created_at": rec.created_at.strftime("%Y-%m-%d %H:%M"),
            }
            flash("Prediction saved in History.", "success")
        except Exception as e:
            flash(f"Prediction failed: {e}", "error")

    return render_template("predict_numerical.html", title="Numerical Prediction", result=result)


def _run_image_prediction(modality):
    is_mri = (modality == "mri")
    model = load_mri_model() if is_mri else load_ct_model()
    class_names = MRI_CLASS_NAMES if is_mri else CT_CLASS_NAMES
    img_size = MRI_IMG_SIZE if is_mri else CT_IMG_SIZE
    label = "MRI" if is_mri else "CT"

    if model is None:
        flash(f"{label} model not found in models/ folder.", "warning")
    img_result = None
    if request.method == "POST":
        if model is None:
            flash(f"Cannot predict: {label} model missing.", "error")
            return redirect(url_for(f"predict_image_{modality}"))
        if "image_file" not in request.files:
            flash("No file uploaded.", "error")
            return redirect(url_for(f"predict_image_{modality}"))
        f = request.files["image_file"]
        if f.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for(f"predict_image_{modality}"))
        if not allowed_file(f.filename):
            flash("Only PNG/JPG/JPEG/DCM/NII/NII.GZ allowed.", "error")
            return redirect(url_for(f"predict_image_{modality}"))
        filename = secure_filename(f.filename)
        if not filename or "." not in filename:
            ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else "png"
            filename = f"scan_{uuid.uuid4().hex[:8]}.{ext}"
        unique = f"{modality}_{uuid.uuid4().hex}_{filename}"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
        f.save(upload_path)
        try:
            from image_preprocessor import (
                load_raw_image,
                preprocess_real_world_scan,
                prepare_model_tensor,
                check_scan_quality,
                validate_brain_scan,
                _is_nifti_file,
            )
            is_nifti = _is_nifti_file(upload_path)
            if is_mri:
                # For MRI keep native contrast as in training
                preprocessed_rgb, raw_rgb = preprocess_real_world_scan(
                    upload_path, target_size=img_size, apply_contrast_enhancement=False
                )
            else:
                # For CT scans: use raw image resized directly to target_size (224x224)
                # without real-world head cropping, CLAHE, or letterboxing, matching
                # the exact native training pipeline of stroke_cnn_model_v2.keras
                raw_rgb = load_raw_image(upload_path)
                preprocessed_rgb = cv2.resize(raw_rgb, img_size)

            # Strict Brain CT / MRI Scan Validation
            validation = validate_brain_scan(
                upload_path,
                raw_rgb,
                expected_modality=modality,
                original_filename=f.filename,
            )
            if not validation["is_valid"]:
                try:
                    os.remove(upload_path)
                except OSError:
                    pass
                flash(validation["error"], "error")
                return redirect(url_for(f"predict_image_{modality}"))

            quality_info = check_scan_quality(preprocessed_rgb)
            for warning_msg in quality_info.get("warnings", []):
                flash(f"Quality Warning: {warning_msg}", "warning")

            display_filename = unique
            if unique.lower().endswith((".dcm", ".nii", ".nii.gz")):
                display_filename = f"{unique}.png"
                display_path = os.path.join(app.config["UPLOAD_FOLDER"], display_filename)
                Image.fromarray(preprocessed_rgb).save(display_path)

            x = prepare_model_tensor(model, preprocessed_rgb, target_size=img_size)
            preds_raw = model.predict(x, verbose=0)
            preds_raw = preds_raw[0] if isinstance(preds_raw, (list, tuple)) else preds_raw
            preds = preds_raw[0]
            pred_idx = int(np.argmax(preds))
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
            overlay, _ = make_gradcam_overlay(preprocessed_rgb, model, img_size, pred_idx)
            gradcam_name = f"gradcam_{modality}_{uuid.uuid4().hex}.png"
            gradcam_path = os.path.join(app.config["OUTPUT_FOLDER"], gradcam_name)
            Image.fromarray(overlay).save(gradcam_path)

            # ------------------------------------------------------------------
            # STEP 4: Conditional Trigger — If MRI prediction is Ischemic,
            # activate Stage-2 U-Net++ Lesion Segmentation (best_stroke_model.keras)
            # ------------------------------------------------------------------
            lesion_result = None
            if is_mri and pred_label.strip().lower() == "ischemic":
                lesion_model = load_lesion_model()
                if lesion_model is not None:
                    lesion_uid = uuid.uuid4().hex[:10]
                    if is_nifti:
                        from lesion_segmenter import run_nifti_lesion_segmentation
                        lesion_result = run_nifti_lesion_segmentation(
                            nifti_path=upload_path,
                            lesion_model=lesion_model,
                            output_folder=app.config["OUTPUT_FOLDER"],
                            unique_id=lesion_uid,
                            prob_thresh=0.5,
                            target_size=img_size,
                        )
                    else:
                        from lesion_segmenter import run_ischemic_lesion_segmentation
                        seq_mode = request.form.get("sequence_mode", "auto")
                        lesion_result = run_ischemic_lesion_segmentation(
                            img_rgb=preprocessed_rgb,
                            lesion_model=lesion_model,
                            output_folder=app.config["OUTPUT_FOLDER"],
                            unique_id=lesion_uid,
                            sequence_mode=seq_mode,
                            prob_thresh=0.5,
                        )
                else:
                    flash(
                        "Stage-2 Lesion Segmentation model (best_stroke_model.keras) not found in models/.",
                        "warning",
                    )

            # ------------------------------------------------------------------
            # Conditional Trigger — If CT prediction is Hemorrhage, activate
            # NeuroFocus AI Stage-2 (5-Subtype ResNet-50 + U-Net ResNet-34 Segmenter)
            # ------------------------------------------------------------------
            ct_hemorrhage_result = None
            if (not is_mri) and pred_label.strip().lower() == "hemorrhage":
                ct_cls_model, ct_seg_model = load_ct_neurofocus_models()
                if ct_cls_model is not None and ct_seg_model is not None:
                    from ct_hemorrhage_analyzer import run_ct_hemorrhage_analysis
                    hem_uid = uuid.uuid4().hex[:10]
                    gt_mask_upload_path = None
                    if "gt_mask_file" in request.files:
                        mf = request.files["gt_mask_file"]
                        if mf and mf.filename:
                            m_unique = f"gt_{hem_uid}_{secure_filename(mf.filename)}"
                            gt_mask_upload_path = os.path.join(app.config["UPLOAD_FOLDER"], m_unique)
                            mf.save(gt_mask_upload_path)

                    ct_hemorrhage_result = run_ct_hemorrhage_analysis(
                        image_path_or_rgb=raw_rgb,
                        classifier=ct_cls_model,
                        segmenter=ct_seg_model,
                        output_folder=app.config["OUTPUT_FOLDER"],
                        unique_id=hem_uid,
                        cls_threshold=0.50,
                        seg_threshold=0.55,
                        gt_mask_path=gt_mask_upload_path,
                    )
                else:
                    flash(
                        "Stage-2 NeuroFocus AI CT models (classifier_best.pth / unet_resnet34_ctich_best.pth) not found in models/.",
                        "warning",
                    )

            rec = PredictionHistory(
                user_id=current_user.id,
                pred_type=f"image_{modality}",
                label=pred_label,
                proba=float(preds[pred_idx]),
                image_filename=display_filename,
                gradcam_filename=gradcam_name
            )
            db.session.add(rec)
            db.session.commit()
            probas = [(class_names[i] if i < len(class_names) else f"Class {i}", float(p)) for i, p in enumerate(preds.tolist())]

            img_result = {
                "pred_label": pred_label,
                "upload_filename": display_filename,
                "gradcam_filename": gradcam_name,
                "probas": probas,
                "lesion_result": lesion_result,
                "ct_hemorrhage_result": ct_hemorrhage_result,
            }
            flash(f"{label} prediction saved in History.", "success")
        except Exception as e:
            flash(f"{label} prediction failed: {e}", "error")

    return render_template(f"predict_image_{modality}.html", title=f"{label} Prediction", img_result=img_result)
    

@app.route("/predict/image", methods=["GET"])
@login_required
def predict_image():
    return render_template("predict_image_select.html", title="Image Prediction")

@app.route("/predict/image-mri", methods=["GET", "POST"])
@login_required
def predict_image_mri():
    return _run_image_prediction("mri")

@app.route("/predict/image-ct", methods=["GET", "POST"])
@login_required
def predict_image_ct():
    return _run_image_prediction("ct")


@app.route("/history")
@login_required
def history():
    rows = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.created_at.desc()).all()
    return render_template("history.html", title="History", rows=rows)


@app.route("/report")
@login_required
def report():
    rows = PredictionHistory.query.filter_by(user_id=current_user.id).all()

    total = len(rows)
    numerical = sum(1 for r in rows if r.pred_type == "numerical")
    mri = sum(1 for r in rows if r.pred_type == "image_mri")
    ct = sum(1 for r in rows if r.pred_type == "image_ct")
    old = sum(1 for r in rows if r.pred_type == "image")
    image = mri + ct + old

    label_counts = {}
    for r in rows:
        if r.label:
            label_counts[r.label] = label_counts.get(r.label, 0) + 1
         # ... label_counts same ...
    stats = {"total": total, "numerical": numerical, "image": image, "mri": mri, "ct": ct}
    chart = {
        "type_data": {"labels": ["Numerical", "MRI", "CT"], "values": [numerical, mri, ct]},
        "label_data": {"labels": list(label_counts.keys()) or ["No Data"], "values": list(label_counts.values()) or [0]},
    }

    return render_template("report.html", title="Report", stats=stats, chart=chart)


@app.route("/api/clinical-qa", methods=["POST"])
@login_required
def api_clinical_qa():
    """Interactive Clinical QA endpoint routing to AHA/ASA Knowledge Base RAG + Google/Web Search RAG."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    patient_context = (data.get("patient_context") or "").strip()
    search_mode = (data.get("search_mode") or "auto").strip()

    if not question:
        return jsonify({"error": "Please enter a clinical question."}), 400

    result = ask_clinical_qa(
        question=question,
        patient_context=patient_context,
        search_mode=search_mode,
    )
    return jsonify(result), 200


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    app.run(debug=True)
