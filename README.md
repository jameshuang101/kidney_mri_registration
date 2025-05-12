# Usage
1. Clone repository and install package
```
pip install .
```

2. Train model
```
kidney-mri-train-affine --data_dir /path/to/dicom_folders --epochs 30 --out affine.pth
kidney-mri-train-deformable --data_dir /path/to/dicom_folders --epochs 30 --out deformable.pth
```

3. Predict & Visualize
```
kidney-mri-run \
  --input /path/to/test_series \
  --affine affine.pth \
  --deformable deformable.pth \
  --output registered.npy \
  --slice 10
```

4. Evaluate
```
kidney-mri-evaluate --fixed /path/to/fixed_series --registered registered.npy
```

# Background & Motivation
**Problem**: Respiratory motion during 4D Discrete Contrast-Enhanced MRI (DCE-MRI) corrupts kidney images, biasing downstream kinetic analyses (e.g., blood flow, GFR).

**Prior Work**: Traditional affine/non-rigid methods (optical flow, demons) can reduce motion but may alter intensity signals and are slow. Deep networks (VoxelMorph, Quicksilver) excel in brain, liver, lung, but have not been applied to abdominal DCE-MRI.

**Goal**: Develop a two-stage, unsupervised deep-learning pipeline (affine → deformable) to correct kidney motion in 4D DCE-MRI, improving both speed and registration fidelity.

# Data & Preprocessing
**Dataset**: DCE-MRI from 20 patients, each with 39 dynamic phases (pre-contrast through nephrographic), acquired on Philips 3T.

**Preprocessing**:
* Manual kidney segmentation on each slice (AnalyzePro).

* Crop to kidney region, convert to 8-bit, resize to 224×384 (bilinear).

* Split by patient: 12 train / 4 val / 4 test (≈19,500 2D images total).

# Two-Stage Registration Pipeline
### Affine Registration Network
![image](https://github.com/user-attachments/assets/24189e85-4c49-4471-9b47-842f0768189c)

* **Architecture**: 2D CNN encoder (five 3×3 conv + max-pool), outputs 6 affine parameters (translation, rotation, scale, shear), then a spatial transformer decoder applies the warp.

* **Loss**: Image similarity: MSE (α = 1)

* **Purpose**: Globally align moving and reference slices to reduce large rigid shifts.

### Deformable Registration Network
![image](https://github.com/user-attachments/assets/b2154399-6adf-4579-88cd-f33adece7286)

* **Architecture**: Adapted from VoxelMorph’s U-Net: encoder (3×3 conv + pooling), bottleneck, decoder (deconv + skip-connections), outputs a Dense Displacement Field (DVF).

* **Spatial Transformer**: Applies DVF to warp both image and segmentation via bilinear sampling.

* **Loss**:
  1. Image similarity (MSE; α = 1)

  2. Segmentation similarity (Sørensen; β = 0.15)

  3. Smoothness regularizer (squared gradient of DVF; λ = 0.05)

# Training & Evaluation
* **Training**: Batch size of 64; 100 epochs; 25 steps/epoch; ~2 hrs total on 8× GTX Titan XPs.

* **Validation**: Monitor NCC, SSD, MSE on held-out set.

* **Testing Metrics**:
  1. Target Registration Error (TRE) at anatomical landmarks

  2. Dice Similarity Coefficient (DSC) & Hausdorff Distance (HD) on segmentation masks

  3. Dynamic Intensity Curves (TICs) within ROI (cortex, medulla)

  4. Visual inspection & subtraction images

# Key Results
| Metric / Comparison     |  Original | Post-Affine | Post-Deformable |
| ----------------------- | :-------: | :---------: | :-------------: |
| **Successive DSC**      |   0.927   |    0.937    |    **0.948**    |
| **Successive HD (mm)**  |    2.96   |     2.68    |     **2.09**    |
| **Successive TRE (mm)** | 3.09±2.51 |  3.04±1.76  |  **2.15±1.34**  |
| **Static DSC**          |   0.928   |    0.933    |    **0.949**    |
| **Static HD (mm)**      |    2.97   |     2.61    |     **2.40**    |
| **Static TRE (mm)**     | 3.18±2.58 |  2.82±2.06  |  **1.09±1.39**  |

![image](https://github.com/user-attachments/assets/e735a470-471b-4e7c-9cce-8058ac665bd4)

![image](https://github.com/user-attachments/assets/23c660c6-b9e5-4fa0-88e0-cff7a307af6a)

# Conclusions & Impact
* **Efficacy**: Two-stage deep registration significantly reduces respiratory motion artifacts in kidney DCE-MRI, preserving intensity fidelity.

* **Speed & Scalability**: Once trained, registration is rapid (inference by CNNs) and unsupervised—no ground-truth deformations needed.

* **Applications**: Improved kinetic modeling (blood flow, GFR), planning (surgery, RT), longitudinal renal function monitoring.

* **Future**: Adaptation to other abdominal organs; integration with real-time/interventional MRI workflows.

The work in this project is directly related to the paper: 
James Huang, Junyu Guo, Ivan Pedrosa, Baowei Fei, "Deep learning-based deformable registration of dynamic contrast enhanced MR images of the kidney," Proc. SPIE 12034, Medical Imaging 2022: Image-Guided Procedures, Robotic Interventions, and Modeling, 1203410 (4 April 2022); doi: 10.1117/12.2611768

PDF: [Fei_2022_SPIE_Huang_Kidney_Segmentation.pdf](https://github.com/JamesHuang404/Kidney-MRI-Registration-Project/files/11174716/Fei_2022_SPIE_Huang_Kidney_Segmentation.pdf)

SPIE 2022 Presentation Slides: [JamesHuang_SPIE_2022_Slides_DLRenalRegistration.pdf](https://github.com/JamesHuang404/Kidney-MRI-Registration-Project/files/11174755/JamesHuang_SPIE_2022_Slides_DLRenalRegistration.pdf)

SPIE 2022 Presentation Video: https://youtu.be/2AH4v2KlWR8
