# VS-SDG: Visual Semantic-Guided Single Domain Generalization for 3D Object Detection

<p align="center">
  <b>Training with Visual Semantics, Deploying with LiDAR Only.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-Accepted-brightgreen">
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-orange">
</p>

---

## 🔥 News

- **[2026.09]** 🎉 VS-SDG has been accepted by RAL.

> Publication information will be updated once an official public version becomes available.

---

## 📖 Overview

LiDAR-based 3D object detectors often suffer from significant performance degradation when deployed across different datasets due to domain shifts caused by variations in LiDAR sensor configurations, point-density patterns, object-scale distributions, and scene contexts.

Existing Single Domain Generalization (SDG) methods for 3D object detection mainly improve geometric robustness through point-cloud augmentation or perturbation. However, they typically rely only on source-domain LiDAR observations and lack an external semantic reference that is less dependent on a specific LiDAR sensor.

In this work, we explore a different direction:

> **Can domain-external visual semantics help a LiDAR detector generalize to unseen domains while keeping the deployed detector completely LiDAR-only?**

We propose **VS-SDG**, a **Visual Semantic-Guided Single Domain Generalization** framework for LiDAR-based 3D object detection.

VS-SDG introduces a frozen vision foundation model only during training and transfers visual semantic knowledge to the LiDAR detector through a sequential semantic-guidance pipeline.

Specifically, VS-SDG consists of three main components:

- **PSBT**: Prompt-Guided Semantic-aware BEV Teacher
- **OSD**: Object-level Semantic Distillation
- **DIFE**: Domain-Invariant Feature Enhancement

During inference, the image branch and vision foundation model are completely removed, and the detector operates only on LiDAR point clouds.

<p align="center">
  <b>No Target-Domain Data · Training-only Visual Semantics · LiDAR-only Inference</b>
</p>

---

## ✨ Highlights

- **Source-only Domain Generalization**  
  VS-SDG is trained using only one labeled source domain without accessing any target-domain data during training.

- **Training-only Visual Semantic Guidance**  
  A frozen Grounded-DINO model provides text-conditioned visual semantic cues during training.

- **Prompt-Conditioned Semantic BEV Teacher**  
  Visual semantic responses are transferred from multi-view images into the LiDAR coordinate system and aggregated into a compact semantic BEV representation.

- **Object-Centric Semantic Distillation**  
  High-confidence semantic hotspots are selected to focus cross-modal supervision on foreground-relevant regions and reduce background-induced negative transfer.

- **Domain-Invariant Feature Enhancement**  
  Feature robustness is improved by enforcing perturbation consistency within semantic hotspots.

- **LiDAR-only Deployment**  
  Camera images and vision foundation models are not required during inference.

---

## 🧠 Motivation

LiDAR domain shifts can arise from multiple factors, including:

- different LiDAR beam configurations,
- different point-density distributions,
- different scanning ranges,
- different object-scale statistics,
- and different scene contexts.

Most existing SDG approaches attempt to improve generalization by modifying the geometry of source-domain point clouds.

However, geometric observations are inherently affected by LiDAR sensor characteristics.

In contrast, high-level object semantics such as **car**, **pedestrian**, and **cyclist** are less dependent on a specific LiDAR configuration.

This motivates us to introduce visual foundation model knowledge as a **domain-external semantic reference** during training.

The key idea of VS-SDG is therefore:

> **Learn domain-robust LiDAR representations with the help of visual semantics during training, while preserving a purely LiDAR-based detector during deployment.**

---

## 🏗️ Framework

<p align="center">
  <img src="assets/framework.png" width="95%">
</p>

<p align="center">
  <i>
  Overall training framework of VS-SDG. The visual semantic teacher is used only during training, while inference remains completely LiDAR-only.
  </i>
</p>

The overall semantic-guidance pipeline can be summarized as:

```text
Multi-view Images
        │
        ▼
Frozen Grounded-DINO
        │
        ▼
Prompt-conditioned
Semantic Responses
        │
        ▼
       PSBT
        │
        ▼
Semantic BEV Teacher
        │
        ▼
       OSD
        │
        ▼
Semantic Hotspots
        │
        ▼
       DIFE
        │
        ▼
Domain-Robust LiDAR Features
        │
        ▼
 3D Object Detection
```

During inference:

```text
LiDAR Point Cloud
        │
        ▼
   LiDAR Detector
        │
        ▼
3D Object Detection
```

All image-related teacher components are removed during deployment.

---

# 🔍 Method

## 1. Prompt-Guided Semantic-aware BEV Teacher (PSBT)

Conventional LiDAR detectors mainly learn geometric representations from point clouds and lack explicit domain-external semantic guidance.

To address this limitation, we introduce a **Prompt-Guided Semantic-aware BEV Teacher (PSBT)**.

We employ a frozen **Grounded-DINO** model as a training-time visual semantic teacher.

Given multi-view images and category prompts such as:

```text
car
pedestrian
cyclist
```

Grounded-DINO produces category-conditioned visual semantic responses.

The obtained semantic responses are transferred from the image space into 3D space through point-to-image projection.

For points belonging to the same BEV pillar, the aligned semantic responses are further aggregated using a lightweight self-attention module.

This produces a compact prompt-conditioned semantic BEV representation that serves as the semantic teacher for the LiDAR detector.

Unlike directly distilling high-dimensional visual foundation model features, VS-SDG transfers category-aware semantic responses into a compact BEV semantic space.

---

## 2. Object-level Semantic Distillation (OSD)

Although PSBT provides dense semantic responses, direct global BEV alignment may introduce unreliable supervision.

Potential noise may originate from:

- background activations,
- point-to-image projection errors,
- camera-LiDAR calibration noise,
- occlusion,
- and view aggregation errors.

To alleviate these problems, we introduce **Object-level Semantic Distillation (OSD)**.

OSD extracts high-confidence local peaks from the semantic BEV teacher and constructs foreground-like semantic hotspot regions.

Instead of enforcing semantic alignment uniformly across the entire BEV feature map, semantic distillation is concentrated on these foreground-relevant regions.

This object-centric design helps reduce background-induced negative transfer while preserving discriminative semantic information around potential objects.

The selected hotspot regions are also reused by DIFE for subsequent feature robustness learning.

---

## 3. Domain-Invariant Feature Enhancement (DIFE)

Cross-domain LiDAR representations can be sensitive to changes in sensor configuration and point-density patterns.

To improve feature robustness, we introduce **Domain-Invariant Feature Enhancement (DIFE)**.

DIFE first applies lightweight channel-adaptive weighting to the LiDAR BEV feature.

Feature-space perturbations are then introduced during training.

Instead of enforcing perturbation consistency across the full BEV space, VS-SDG performs consistency regularization only inside the semantic hotspot regions selected by OSD.

The model is encouraged to preserve local feature relations before and after perturbation.

This design focuses domain-invariant learning on foreground-relevant regions while reducing interference from background domain shifts.

---

## 4. Overall Objective

VS-SDG jointly optimizes the original 3D detection objective together with semantic distillation and domain-invariant regularization:

```text
L_total = L_det + λ_sem L_sem + λ_obj L_obj + λ_dife L_DIFE
```

where:

- `L_det` denotes the original 3D detection loss,
- `L_sem` denotes semantic BEV distillation,
- `L_obj` denotes object-level semantic distillation,
- `L_DIFE` denotes hotspot-constrained perturbation consistency.

The visual teacher remains frozen throughout training.

---

# 📊 Main Results

We evaluate VS-SDG under three cross-dataset Single Domain Generalization settings:

- **NuScenes → KITTI**
- **Waymo → NuScenes**
- **Waymo → KITTI**

The detector is trained only on the source domain and directly evaluated on the unseen target domain.

**No target-domain data are used during training.**

Values denote **APBEV / AP3D**.

---

## NuScenes → KITTI

| Method | Car | Pedestrian | Cyclist | mAP |
|:---|---:|---:|---:|---:|
| Source-only | 62.69 / 19.02 | 22.72 / 18.37 | 20.61 / 18.13 | 35.34 / 18.50 |
| PA-DA | 65.09 / 32.44 | 18.73 / 14.94 | 18.66 / 15.91 | 34.16 / 21.10 |
| 3D-VF | 65.36 / 29.21 | 24.85 / 20.87 | 22.13 / 19.31 | 37.45 / 23.13 |
| PDDA | 73.58 / 33.11 | 30.01 / 23.73 | 22.93 / 18.62 | 42.17 / 25.15 |
| **VS-SDG** | **76.40 / 35.52** | **32.18 / 25.82** | **24.76 / 20.04** | **44.45 / 27.13** |

---

## Waymo → NuScenes

| Method | Car | Pedestrian | Cyclist | mAP |
|:---|---:|---:|---:|---:|
| Source-only | 31.20 / 19.13 | 10.52 / 8.39 | 0.75 / 0.55 | 14.16 / 9.36 |
| PA-DA | 29.43 / 18.06 | 10.84 / 8.43 | 0.82 / 0.43 | 13.70 / 8.97 |
| 3D-VF | 30.17 / 18.91 | 10.54 / 7.23 | 0.76 / 0.78 | 13.82 / 8.97 |
| PDDA | 36.04 / 22.25 | 14.48 / 10.56 | 1.15 / 0.95 | 17.22 / 11.26 |
| **VS-SDG** | **38.77 / 23.91** | **15.92 / 11.80** | **1.42 / 1.18** | **18.70 / 12.30** |

---

## Waymo → KITTI

| Method | Car | Pedestrian | Cyclist | mAP |
|:---|---:|---:|---:|---:|
| Source-only | 66.65 / 19.27 | 66.55 / 64.00 | 63.04 / 57.11 | 65.41 / 46.79 |
| PA-DA | 65.82 / 17.61 | 66.40 / 63.88 | 61.30 / 56.23 | 64.51 / 45.91 |
| 3D-VF | 66.72 / 19.37 | 66.21 / 63.12 | 62.74 / 56.44 | 65.22 / 46.31 |
| PDDA | 69.90 / 20.21 | 63.24 / 62.59 | 63.27 / 57.21 | 65.47 / 46.67 |
| **VS-SDG** | **72.83 / 22.18** | **67.42 / 64.23** | **64.90 / 58.44** | **68.38 / 48.28** |

---

# 🧪 Component Ablation

We progressively integrate PSBT, OSD, and DIFE to analyze their individual contributions.

| PSBT | OSD | DIFE | NuScenes → KITTI | Waymo → NuScenes |
|:---:|:---:|:---:|---:|---:|
|  |  |  | 35.34 / 18.50 | 14.15 / 9.36 |
| ✓ |  |  | 40.02 / 23.00 | 16.13 / 10.65 |
| ✓ | ✓ |  | 42.63 / 25.22 | 17.47 / 11.55 |
| ✓ | ✓ | ✓ | **44.45 / 27.13** | **18.70 / 12.30** |

The three components form a sequential semantic-guidance pipeline:

```text
PSBT
  │
  ▼
Semantic BEV Teacher
  │
  ▼
OSD
  │
  ▼
Foreground Semantic Hotspots
  │
  ▼
DIFE
  │
  ▼
Domain-Robust LiDAR Representation
```

The results show that each component provides complementary improvements to cross-domain 3D detection.

---

# 🎯 Comparison of Distillation Strategies

We further compare different semantic distillation strategies.

| Distillation Strategy | NuScenes → KITTI | Waymo → NuScenes |
|:---|---:|---:|
| Feature-level Distillation | 40.12 / 22.77 | 15.43 / 8.91 |
| GT-guided Instance-level Distillation | 41.81 / 24.26 | 15.78 / 9.11 |
| **OSD Hotspot Distillation** | **44.45 / 27.13** | **18.70 / 12.30** |

Global feature-level distillation may introduce excessive background supervision.

GT-guided instance-level distillation provides more localized supervision but depends on predefined object regions.

In contrast, OSD automatically identifies semantic hotspots from the visual semantic teacher and achieves stronger cross-domain performance.

---

# 🔄 Feature Robustness Analysis

We investigate different regions for feature consistency regularization.

| Consistency Region | NuScenes → KITTI | Waymo → NuScenes |
|:---|---:|---:|
| w/o DIFE Consistency | 42.63 / 25.22 | 17.47 / 11.55 |
| Full-BEV Consistency | 43.21 / 25.78 | 17.82 / 11.84 |
| Random-mask Consistency | 43.58 / 26.12 | 18.03 / 11.98 |
| **OSD-hotspot Consistency** | **44.45 / 27.13** | **18.70 / 12.30** |

The results demonstrate that feature consistency is more effective when applied to foreground-relevant semantic regions rather than the full BEV feature map.

---

# 🧩 Semantic Feature Aggregation

Different strategies for aggregating point-aligned semantic responses are compared below.

| Aggregation Strategy | NuScenes → KITTI | Waymo → NuScenes |
|:---|---:|---:|
| Max-pooling | 40.92 / 23.13 | 16.71 / 10.33 |
| Average-pooling | 41.36 / 25.61 | 15.54 / 10.12 |
| **Self-attention-based Aggregation** | **44.45 / 27.13** | **18.70 / 12.30** |

Self-attention-based aggregation provides a better balance between semantic discrimination and robustness against noisy point-to-image responses.

---

# 📈 Grounded-DINO Feature Level

We analyze semantic features extracted from different levels of Grounded-DINO on the Waymo → KITTI setting.

| PSBT Feature Level | mAPBEV | mAP3D |
|:---|---:|---:|
| Level 1 (Shallow) | 62.12 | 41.35 |
| Level 2 (Middle) | 64.43 | 42.12 |
| **Level 3 (Deep)** | **68.38** | **48.28** |

Deeper visual foundation model features provide more object-centric semantic responses and stronger domain-generalization performance.

---

# ⚡ Inference Efficiency

An important property of VS-SDG is that the vision foundation model is used **only during training**.

| Method | Params | Latency | FPS | Memory |
|:---|---:|---:|---:|---:|
| VoxelRCNN | 16.75 M | 108.23 ms | 9.24 | 3.18 GB |
| **VS-SDG** | **16.92 M** | **109.89 ms** | **9.10** | **3.24 GB** |

During inference, the following components are removed:

- Multi-view camera images
- Grounded-DINO
- PSBT teacher branch
- OSD
- Semantic distillation losses
- Perturbation-consistency loss

Only the LiDAR detector and the lightweight DIFE channel-weighting module are retained.

Therefore, VS-SDG introduces only minor additional deployment overhead compared with the original VoxelRCNN detector.

---

# 🖼️ Visualization

## Cross-Domain Detection

<p align="center">
  <img src="assets/detection_visualization.png" width="95%">
</p>

<p align="center">
  <i>
  Qualitative cross-domain 3D detection results.
  </i>
</p>

---

## Semantic Feature Visualization

<p align="center">
  <img src="assets/feature_visualization.png" width="90%">
</p>

<p align="center">
  <i>
  Visualization of LiDAR BEV feature responses before and after visual semantic guidance.
  </i>
</p>

---

## Feature Distribution

<p align="center">
  <img src="assets/tsne.png" width="75%">
</p>

<p align="center">
  <i>
  t-SNE visualization of cross-domain feature distributions.
  </i>
</p>

---

# 📦 Code Release

🚧 **The implementation of VS-SDG is currently being organized and will be released in this repository.**

The future code release will include:

- VS-SDG implementation
- PSBT implementation
- OSD implementation
- DIFE implementation
- Configuration files
- Dataset preparation instructions
- Training scripts
- Evaluation scripts
- Cross-domain evaluation settings

> **Note:** Pretrained model weights will **not** be provided.  
> All reported results can be reproduced by training the released implementation following the provided configurations.


---

# 🙏 Acknowledgement

This project is built upon several excellent open-source projects and research works.

We sincerely thank the authors and contributors of:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for providing the open-source 3D object detection framework.
- [Voxel R-CNN](https://github.com/djiajunustc/Voxel-R-CNN) for the strong LiDAR-based 3D object detection baseline.
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for the open-set vision-language grounding model used as the training-time visual semantic teacher.

We also thank the authors of the related domain generalization and cross-modal knowledge transfer methods that inspired this work.

If you find these projects useful, please consider citing their original papers as well.

---

# 📄 License

The license information will be provided together with the official code release.

Please also follow the licenses of the third-party projects used in this repository, including OpenPCDet, Voxel R-CNN, and Grounding DINO.

