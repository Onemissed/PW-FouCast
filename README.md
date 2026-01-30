# Extending Precipitation Nowcasting Horizons via Spectral Fusion of Radar Observations and Foundation Model Priors

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official implementation of **PW-FouCast**, a framework that integrates atmospheric foundation model priors with high-resolution radar observations for superior precipitation nowcasting.

---

## 🌟 Key Features

> **PW-FouCast:** Pangu-Weather-guided Fourier-domain foreCast. 
> Our model leverages spectral fusion to bridge the gap between large-scale atmospheric dynamics and local-scale convective patterns.

### 🏗️ Model Architecture
[//]: # (![Model Architecture]&#40;docs/assets/model_architecture.png&#41;)
<p align="center">
  <img src="docs/assets/model_architecture.png" width="1000" alt="Model Architecture">
</p>

[//]: # (*Figure 1: Overview of the PW-FouCast framework.*)
<p align="center">
  <em>Figure 1: Overview of the PW-FouCast framework.</em>
</p>

---

## 📊 Implemented Models

| Category                 | Models                                                       |
|:-------------------------|:-------------------------------------------------------------|
| **Proposed**             | **PW-FouCast**                                               |
| **Unimodal Baselines**   | PredRNN v2, SimVP v2, TAU, Earthformer, PastNet, AlphaPre, NowcastNet, LMC-Memory, AFNO |
| **Multimodal Baselines** | LightNet, MM-RNN, CM-STjointNet |

---

## 📂 Repository Structure

```text
├── config/             # YAML configurations (MeteoNet & SEVIR)
├── data_index/         # Dataset indexing and manifest files
├── evaluation/         # Metrics (CSI, HSS, MSE) and evaluation scripts
├── model/              # Implementation of PW-FouCast and baselines
├── module/             # Shared building blocks (convolutions, attention, etc.)
├── util/               # Logging, visualization, and utility functions
└── README.md           # Project documentation

```

---

## 📥 Dataset Preparation

We evaluate our method on two primary datasets: **SEVIR-LR** and **MeteoNet**.

### 1. MeteoNet Dataset

1. **Download:** [MeteoNet Radar Reflectivity](https://meteonet.umr-cnrm.fr/dataset/data/NW/radar/reflectivity_old_product/)
2. **Process:**
```bash
# Convert raw .npz to .npy and downsample
python save_meteonet.py

# Partition into events using sliding windows
python split_meteonet.py

```



### 2. SEVIR-LR Dataset

1. **Download:** [SEVIR-LR Dataset Link](https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip)
2. **Process:**
```bash
python process_sevir.py  # .h5 to .npy
python save_sevir.py     # Split into single events
python split_sevir.py    # Sliding window partition

```


---

## 🏃 Quick Start

### Training

To train a model (e.g., AFNO) on your chosen dataset, use the following commands:

**SEVIR-LR:**

```bash
python train_baseline_sevir.py --model afno --batchsize 16 --epoch 100 --lr 1e-3 --gpus 0

```

**MeteoNet:**

```bash
python train_meteonet.py --model afno --batchsize 16 --epoch 100 --lr 1e-3 --gpus 0

```

---

## 📈 Experimental Results

*Figure 2: Qualitative comparison of PW-FouCast against SOTA baselines on a heavy precipitation event.*

| Model | CSI (0.5) | HSS | MSE ↓ |
| --- | --- | --- | --- |
| **PW-FouCast** | **0.XXX** | **0.XXX** | **X.XX** |
| Earthformer | 0.XXX | 0.XXX | X.XX |
| NowcastNet | 0.XXX | 0.XXX | X.XX |

