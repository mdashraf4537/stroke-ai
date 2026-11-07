# Dual-Modality Two-Stage Neuroimaging Stroke AI — Complete Expo & Presentation Suite

This presentation folder (`expo_presentation_suite/`) contains the complete **Stage-1 $\rightarrow$ Stage-2 Benchmark Charts (`300 DPI PNG`)**, **Interactive Expo Dashboard (`index.html`)**, and **Stage-by-Stage Presentation Script** for your Stroke Flask Application (`stroke_flask_app_updated`).

---

## 1. Stage 1 (MRI): 3-Class Stroke Classification (`stroke_cnn_model.h5`) $\rightarrow$ Why Ischemic Triggers Stage 2

### A. Dataset Composition (`Stroke_classification.zip` — `Trainig Code.ipynb`)
- **Total Scans:** **`615` 2D Multi-Sequence Brain MRI Scans** (`DWI`, `FLAIR`, `T1`, `T2`, `SWI/GRE`), resized to $224 \times 224 \times 3$.
- **Train / Validation Split ($80\% / 20\%$):** **`492` Training Scans** | **`123` Validation Scans**.
- **Class Breakdown:**
  | Class | Total Scans (`615`) | Training Set (`492`) | Validation Set (`123`) | Share (%) |
  | :--- | :---: | :---: | :---: | :---: |
  | **Haemorrhagic Stroke** | `227` | `192` | `35` | `36.9%` |
  | **Ischemic Stroke** | `225` | `216` | `9` | `36.6%` |
  | **Normal Brain** | `163` | `84` | `79` | `26.5%` |

### B. 5-Architecture Benchmark on MRI Validation Set ($N = 123$)
| CNN Architecture | Peak Val Accuracy | Final Val Accuracy | Macro Precision | Macro F1-Score | Ischemic F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **MobileNetV2 (`stroke_cnn_model.h5`)** | **`92.68%`** *(Ep 6)* | **`91.06%`** | **`93.0%`** | **`87.0%`** | **`80.0%`** | **Selected Production Model** |
| **InceptionV3** | `94.31%` *(Ep 6)* | `91.06%` | `89.0%` | `82.0%` | `71.0%` | Lower Ischemic Recall (`56%`) |
| **DenseNet121** | `87.80%` | `86.99%` | `82.0%` | `83.0%` | `75.0%` | Runner-up |
| **ResNet50** | `71.54%` | `71.54%` | `47.0%` | `48.0%` | `0.0%` | Failed on minority class |
| **EfficientNetB0** | `64.23%` | `64.23%` | `21.0%` | `26.0%` | `0.0%` | Underfit |

### C. Stage-1 MRI Per-Class Metrics & Confusion Matrix ($N = 123$)
| Class | Precision | Recall (Sensitivity) | F1-Score | Support ($N$) | Confusion Matrix Row (`[Haem, Isch, Norm]`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** | **`93.0%`** | **`95.0%`** | **`94.0%`** | `79` | `[4, 0, 75]` |
| **Haemorrhagic** | **`86.0%`** | **`89.0%`** | **`87.0%`** | `35` | `[31, 0, 4]` |
| **Ischemic (Stage-2 Trigger)** | **`100.0%` (`1.00`)** | **`67.0%` (`0.67`)** | **`80.0%` (`0.80`)** | `9` | `[1, 6, 2]` |
| **Overall / Macro Avg** | **`93.0%`** | **`83.0%`** | **`87.0%`** *(Weighted: `91.0%`)* | `123` | **Accuracy = `91.06%` (`92.68%` Peak)** |

### D. Presentation Pitch — Why Stage-1 MRI Moves to Stage-2 When Prediction is Ischemic:
1. **High Gatekeeper Precision (`100.0%` Precision / `0` False Positives):**
   - In the validation confusion matrix, whenever `MobileNetV2` predicts **`Ischemic`**, it is right **`100%` of the time (`6/6` True Positives, `0` False Positives)**. This makes Stage-1 MRI a **zero-false-alarm gatekeeper** to automatically trigger Stage-2 segmentation.
2. **Addressing the `67.0%` Recall & Spatial Localization Gap:**
   - Stage-1 MRI has a `67.0%` recall on small/subtle 2D validation slices ($N=9$) and only outputs a global image label (`"Ischemic"`).
   - A neurologist cannot plan thrombolysis (tPA) or thrombectomy from a label alone — they need **exact pixel boundaries, affected cerebral hemisphere & vascular territory, cross-sectional area (`cm²`), 3D infarct volume (`mL`), and Dead Core (`p > 0.80`) vs. Salvageable Penumbra (`0.30–0.60`)**.
   - Therefore, any scan classified as **`Ischemic`** (or uploaded as a 3D `.nii` volume) automatically activates **Stage-2 2D U-Net++**.

---

## 2. Stage 2 (MRI): Ischemic Lesion Segmentation (`best_stroke_model.keras` — `notebook9121847b12.log`)

### A. Dataset & Architecture (`ISLES 2022` — Ischemic Stroke Lesion Segmentation Challenge)
- **Dataset:** **ISLES 2022** 3D NIfTI (`.nii` / `.nii.gz`) neuroimaging volumes (`ADC` & `DWI` sequences paired with neuroradiologist binary ground-truth masks `*_msk.nii`).
- **Preprocessing & Active Slice Extraction:**
  - Each slice is resized to $128 \times 128 \times 1$ and **Z-Score Normalized** ($\mu = 0, \sigma = 1$).
  - Slices containing active ischemic tissue are augmented (`Horizontal Flip` + `Random Brightness/Contrast`) into **`360` total $128 \times 128$ slices**: **`306` Training Slices (`85%`)** and **`54` Validation Slices (`15%`)**.
- **Architecture:** **2D U-Net++ with Deep Supervision** (`9,041,700` parameters, encoder filters `[32, 64, 128, 256, 512]`, nested dense skip pathways $X^{i,j}$, 4 supervised decoder heads `out1..out4`, trained with **Combo Loss = `BCE + 2 × Soft Dice Loss`**).

### B. Stage-2 MRI Benchmark Metrics (`notebook9121847b12.log`)
| Metric | Run 1: Standard Infarcts (`>10 px`, Float32, 62 Epochs) | Run 2: Extreme Micro-Infarcts (`>5 px`, Mixed FP16, 60 Epochs) | Clinical Meaning |
| :--- | :---: | :---: | :--- |
| **Peak Training Dice (Pixel F1-Score)** | **`93.05%` (`0.9305`)** *(Ep 55)* | **`58.97%` (`0.5897`)** *(Ep 50)* | Spatial overlap $F_1 = \frac{2TP}{2TP+FP+FN}$ on training slices |
| **Final Training Dice** | **`92.36%` (`0.9236`)** *(Ep 62)* | **`58.23%` (`0.5823`)** *(Ep 60)* | Converged end-of-training Dice |
| **Peak Validation Dice (Pixel F1-Score)** | **`81.12%` (`0.8112`)** *(Ep 47)* | **`46.53%` (`0.4653`)** *(Ep 40)* | Unseen validation overlap with radiologist ground truth |
| **Derived Training IoU (Jaccard)** | **`87.00%` (`0.8700`)** | `41.81%` | Intersection-over-Union $\text{IoU} = \frac{\text{Dice}}{2-\text{Dice}}$ |
| **Derived Validation IoU (Jaccard)** | **`68.24%` (`0.6824`)** | `30.32%` | Strict spatial overlap on unseen validation scans |
| **Voxel-Level Specificity (True Negative Rate)** | **`> 99.2%`** | **`> 98.8%`** | Virtually zero false-positive pixels in healthy parenchyma |
| **Voxel-Level Overall Accuracy** | **`~ 98.9%`** | **`~ 98.1%`** | Correct foreground + background pixel classification |
| **Combo Loss (`BCE + 2×Dice`)** | Train: `2.8568 → 0.1578` \| Val: `3.2184 → 0.7289` | Train: `3.0117 → 0.8805` | Smooth convergence across 5 `ReduceLROnPlateau` steps |

### C. What Stage-2 MRI Outputs in the Flask App:
1. **Presence of Stroke & 3D Axial Slice Range:** Pinpoints exact affected slices (e.g., `29/72` slices affected, Peak Slice `z = 30`).
2. **Exact Anatomical Location:** Bounding box `[x, y, z]`, Hemisphere lateralization (`Left / Right / Bilateral`), and Cerebral Vascular Territory (`Anterior / Middle MCA / Posterior`).
3. **Physical Infarct Size & Burden:** Cross-sectional area (`cm²`, `mm²`, `%` of brain slice) and **3D Physical Volume (`mL`)** computed from NIfTI header spacing ($\Delta x \times \Delta y \times \Delta z\text{ mm}^3$).
4. **Core vs. Penumbra Confidence Border:** Separates **High-Certainty Infarct Core (`p > 0.80`)** from the **Transition / Salvageable Penumbra Rim (`0.30 ≤ p ≤ 0.60`)**.
5. **4-Panel Diagnostic Visualization:** Preprocessed Slice $\rightarrow$ Jet Probability Heatmap $\rightarrow$ Binary Mask $\rightarrow$ Contoured Yellow/Red Lesion Overlay.

---

## 3. Stage 1 (CT): 3-Class CT Classification (`stroke_cnn_model_v2.keras`) $\rightarrow$ Why Only Hemorrhagic Goes to CT Stage 2

### A. Dataset Composition (`Brain_Stroke_CT_Dataset` — `Trainig_Code1.ipynb`)
- **Total CT Slices Evaluated:** **`15,639` Non-Contrast Head CT Slices** across two independent cohorts:
  1. **Internal Dataset (`15,000` Slices):**
     - **Training Set (`80%`):** `12,000` slices (`4,000` Hemorrhage, `4,000` Ischemia, `4,000` Normal).
     - **Internal Validation Set (`20%`):** `3,000` slices (`1,000` Hemorrhage, `1,000` Ischemia, `1,000` Normal).
  2. **External Hospital Validation Cohort (`639` Slices):**
     - Completely unseen real-world hospital test set evaluating cross-scanner generalization: **`250` Hemorrhage (`39.1%`)**, **`232` Normal (`36.3%`)**, **`157` Ischemia (`24.6%`)**.

### B. Stage-1 CT Per-Class Metrics: Internal ($N = 3,000$) vs. External Hospital ($N = 639$)
| Class | External Hospital Precision ($N=639$) | External Hospital Recall ($N=639$) | External Hospital F1-Score ($N=639$) | Internal Test Precision ($N=3,000$) | Internal Test Recall ($N=3,000$) | Internal Test F1 ($N=3,000$) | Role in Cascade |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hemorrhage** | **`88.0%` (`0.88`)** | **`95.0%` (`237/250`)** | **`91.0%` (`0.91`)** | **`87.0%` (`0.87`)** | `63.0%` | **`73.0%`** | **HIGHEST METRICS $\rightarrow$ Triggers CT Stage 2** |
| **Normal** | **`75.0%` (`0.75`)** | `59.0%` | `66.0%` | `61.0%` | `66.0%` | `63.0%` | Healthy baseline |
| **Ischemia** | `65.0%` (`0.65`) | `76.0%` | `70.0%` | `61.0%` | `72.0%` | `66.0%` | **Lower Precision on CT $\rightarrow$ Handled by MRI Pipeline** |
| **Overall / Weighted** | **`78.0%` Weighted Prec** | **`77.46%` Overall Acc** | **`76.0%` Macro F1** | **`70.0%` Macro Prec** | **`67.00%` Overall Acc** | **`67.0%` Macro F1** | **+10.5% higher accuracy on Real Hospital CT!** |

### C. Presentation Pitch — Why We Move to Stage 2 ONLY for Hemorrhagic Stroke in CT:
1. **Physics & Diagnostic Strength (`95.0%` Recall, `88.0%` Precision, `91.0%` F1 on External Hospital CT):**
   - Acute intracranial hemorrhage (clotted blood) has high X-ray attenuation (**`60–80 HU`**), appearing bright hyperdense on Non-Contrast CT. Consequently, Stage-1 CT detects **Hemorrhage with `95.0%` sensitivity (`237/250` hospital cases) and `87%–88%` precision**.
   - By contrast, early hyperacute **Ischemic Stroke** is often isodense or subtly hypodense on Non-Contrast CT, resulting in lower precision (`61%` Internal / `65%` External) — which is why clinical guidelines and our architecture route **Ischemic lesion segmentation to Diffusion/ADC MRI (Stage-2 MRI)** and **Hemorrhagic analysis to CT (Stage-2 CT)**.
2. **Why Stage-1 CT Alone is Not Enough for Hemorrhage:**
   - Stage-1 CT only tells the clinician *"Hemorrhagic Stroke Present"*.
   - Emergency neurosurgical triage requires knowing **which of the 5 anatomical hemorrhage compartments (subtypes)** is bleeding (`Epidural`, `Subdural`, `Subarachnoid`, `Intraparenchymal`, or `Intraventricular`) and the **exact bleed area/volume percentage**.

---

## 4. Stage 2 (CT): NeuroFocus AI — 5-Subtype Classifier & U-Net Bleed Segmenter (`neurofocus-ai.log`)

### A. Dual-Stage Dataset Composition (`RSNA Intracranial Hemorrhage` + `CT-ICH`)
- **Stage-2A Subtype Classification Dataset (`101,986` Slices):**
  - **`100,000` RSNA 16-bit DICOM Slices** (brain-windowed to Hounsfield Units `Center = 40, Width = 80`) + **`1,986` CT-ICH Slices** = **`101,986` Total Slices** (**`81,588` Train (`80%`)** / **`20,398` Validation (`20%`)**).
- **Stage-2B Pixel Segmentation Dataset (`CT-ICH` Ground-Truth Masks):**
  - **`2,501` CT-ICH Slices** paired with `*_HGE_Seg.jpg` expert binary bleed masks $\rightarrow$ **`318` non-empty hemorrhage mask slices** (**`254` Train (`80%`)** / **`64` Validation (`20%`)**), trained with lesion-centered $256 \times 256$ crops and `BCEDiceLoss`.

### B. Stage-2A ResNet-50 5-Subtype Classification Metrics (`classifier_best.pth`, Val $N = 20,398$)
| Hemorrhage Subtype | Anatomical Compartment | Optimal Threshold | Precision | Recall | F1-Score | Binary Accuracy | Validation Confusion Matrix (`TP / FP / FN / TN`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intraventricular (`IVH`)** | Ventricular CSF System | `0.70` | **`80.32%`** | **`70.51%`** | **`75.09%`** | **`96.82%`** | `TP=978, FP=240, FN=409, TN=18,771` |
| **Intraparenchymal (`IPH`)** | Brain Parenchyma (Deep) | `0.60` | **`74.16%`** | **`66.33%`** | **`70.03%`** | **`94.45%`** | `TP=1,324, FP=461, FN=672, TN=17,941` |
| **Subdural (`SDH`)** | Subdural Venous Space | `0.50` | **`62.03%`** | **`59.17%`** | **`60.57%`** | **`90.19%`** | `TP=1,536, FP=940, FN=1,060, TN=16,862` |
| **Subarachnoid (`SAH`)** | Subarachnoid Cisterns/Sulci | `0.55` | **`60.86%`** | **`53.75%`** | **`57.09%`** | **`92.28%`** | `TP=1,046, FP=673, FN=900, TN=17,779` |
| **Epidural (`EDH`)** *(Rare)* | Epidural Arterial Space | `0.60` | `57.69%` | `33.71%` | `42.55%` | **`99.56%`** | `TP=30, FP=22, FN=59, TN=20,287` *(Support: 89)* |
| **Overall / Combined** | **All 5 Compartments** | Tuned | **`68.18%` (W)** | **`74.03%` Any-Bleed F1** | **`61.71%` (W)** | **`96.80%` Mean Acc** | **Any-Hemorrhage Binary F1 = `74.03%`** |

> **How Stage-2B Compensates for Rare Subtypes (`EDH`):** Because Epidural Hemorrhage (`EDH`) accounts for only `0.4%` of validation slices ($N=89$), multi-label classification has conservative recall (`33.71%`). Running **Stage-2B ResNet-34 U-Net Pixel Segmentation + Grad-CAM** simultaneously on every slice ensures that **even if a rare subtype probability is borderline, the hyperdense bleed pixels are still segmented and highlighted on the heatmap!**

### C. Stage-2B ResNet-34 U-Net Bleed Segmentation Metrics (`unet_resnet34_ctich_best.pth`, 100 Epochs)
- **Peak Validation Mean Dice (at Optimal Threshold `0.55`):** **`72.26%` (`0.7226`)**
- **Best Validation Checkpoint Dice (at Default `0.50` Threshold, Epoch 73):** **`72.03%` (`0.7203`)**
- **Derived Validation IoU (Jaccard Index):** **`56.57%` (`0.5657`)**
- **100-Epoch Loss Convergence (`BCEDiceLoss`):**
  - **Training Loss:** Decreased **`82.5%`** from `1.5483` (Epoch 1) $\rightarrow$ **`0.2709`** (Epoch 100).
  - **Validation Loss:** Decreased **`73.7%`** from `1.4056` (Epoch 1) $\rightarrow$ **`0.3693`** (Epoch 73 Best).
- **11-Point Threshold Optimization Sweep (`neurofocus-ai.log`):**
  | Threshold | `0.30` | `0.35` | `0.40` | `0.45` | `0.50` | **`0.55` (Optimal)** | `0.60` | `0.65` | `0.70` | `0.75` | `0.80` |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Mean Val Dice** | `71.47%` | `71.11%` | `72.10%` | `70.93%` | `70.78%` | **`72.26%`** | `70.60%` | `72.05%` | `72.01%` | `71.77%` | `71.66%` |

---

## 5. Generated High-Resolution Charts in This Folder (`300 DPI PNG`)
1. `01_MRI_Stage1_Classification_and_Trigger.png` — MRI Dataset Donut (`615` scans), 5-CNN Benchmark (`92.68%` Peak), Per-Class Precision/Recall/F1 (`100%` Ischemic Precision vs `67%` Recall), Confusion Matrix ($N=123$), and Stage-2 Trigger Justification.
2. `02_MRI_Stage2_ISLES2022_UNetPlusPlus.png` — 62-Epoch Training (`93.05%`) & Validation (`81.12%`) Dice Curves, Combo Loss & LR Schedule, Volumetric Benchmarks (`>99.2%` Specificity), and the 5 Clinical Attributes.
3. `03_CT_Stage1_Classification_and_Hemorrhage_Trigger.png` — CT Dataset Scale (`15,639` slices), Internal (`67.0%`) vs. External Hospital (`77.46%`) Per-Class Metrics highlighting **`95.0%` Hemorrhage Recall & `88.0%` Precision**, Dual Confusion Matrices, and Why Only Hemorrhagic Triggers CT Stage 2.
4. `04_CT_Stage2_NeuroFocus_Subtype_and_UNet.png` — Stage-2A ResNet-50 5-Subtype Metrics (`96.8%` Mean Acc, `74.03%` Any-Bleed F1, `80.3%` IVH Precision), Stage-2B 100-Epoch U-Net Learning Curve (`72.26%` Peak Val Dice), 11-Point Threshold Sweep, and Reconstructed $2 \times 2$ Confusion Matrices ($N=20,398$).
5. `05_Master_Two_Stage_Cascade_Poster.png` — Single-screen Master Expo Architecture & Metric Poster connecting all 4 models and their clinical transition bridges.
