# Background & Motivation
Problem: Respiratory motion during 3D DCE-MRI corrupts kidney images, biasing downstream kinetic analyses (e.g., blood flow, GFR).

Prior Work: Traditional affine/non-rigid methods (optical flow, demons) can reduce motion but may alter intensity signals and are slow. Deep networks (VoxelMorph, Quicksilver) excel in brain, liver, lung, but have not been applied to abdominal DCE-MRI.

Goal: Develop a two-stage, unsupervised deep-learning pipeline (affine → deformable) to correct kidney motion in 4D DCE-MRI, improving both speed and registration fidelity.

# Data & Preprocessing
Dataset: DCE-MRI from 20 patients, each with 39 dynamic phases (pre-contrast through nephrographic), acquired on Philips 3T.

Preprocessing:
* Manual kidney segmentation on each slice (AnalyzePro).

* Crop to kidney region, convert to 8-bit, resize to 224×384 (bilinear).

* Split by patient: 12 train / 4 val / 4 test (≈19,500 2D images total).

The work in this project is directly related to the paper: 
James Huang, Junyu Guo, Ivan Pedrosa, Baowei Fei, "Deep learning-based deformable registration of dynamic contrast enhanced MR images of the kidney," Proc. SPIE 12034, Medical Imaging 2022: Image-Guided Procedures, Robotic Interventions, and Modeling, 1203410 (4 April 2022); doi: 10.1117/12.2611768

PDF: [Fei_2022_SPIE_Huang_Kidney_Segmentation.pdf](https://github.com/JamesHuang404/Kidney-MRI-Registration-Project/files/11174716/Fei_2022_SPIE_Huang_Kidney_Segmentation.pdf)

SPIE 2022 Presentation Slides: [JamesHuang_SPIE_2022_Slides_DLRenalRegistration.pdf](https://github.com/JamesHuang404/Kidney-MRI-Registration-Project/files/11174755/JamesHuang_SPIE_2022_Slides_DLRenalRegistration.pdf)

SPIE 2022 Presentation Video: https://youtu.be/2AH4v2KlWR8
