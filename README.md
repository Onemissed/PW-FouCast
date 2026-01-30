This repository contains the full codebase for the paper:

> **Extending Precipitation Nowcasting Horizons via Spectral Fusion of Radar Observations and Foundation Model Priors**  

The project implements:

1. **Our proposed model**  
   - Pangu-Weather-guided Fourier-domain foreCast (PW-FouCast)

2. **Baseline models** for precipitation nowcasting:  
   - PredRNN v2
   - SimVP v2
   - TAU  
   - Earthformer  
   - PastNet  
   - AlphaPre  
   - NowcastNet  
   - LMC-Memory  
   - AFNO  
   - LightNet  
   - MM-RNN  
   - CM-STjointNet  

---

## 📂 Repository Structure

```

├── config/                  # YAML config files for each model
│   ├── meteonet/            # Config files for the MeteoNet dataset
│   └── sevir/               # Config files for the SEVIR dataset
├── data_index/              # Dataset indexing
├── evaluation/              # Code for model evaluation and metric computation
├── model/                   # Model implementations
├── module/                  # Core building blocks used across different models
├── util/                    # Utility functions
└── README.md                # This file

````

---


## 📥 Dataset

We use **SEVIR-LR** and **MeteoNet** dataset for training and evaluation:

1. **MeteoNet dataset**:
   * **Download**:
      Visit [https://meteonet.umr-cnrm.fr/dataset/data/NW/radar/reflectivity_old_product/](https://meteonet.umr-cnrm.fr/dataset/data/NW/radar/reflectivity_old_product/)
   
   * **Processing**:
   We provide two helper scripts to convert the raw npz file into NumPy arrays and to split out individual precipitation events
        ```
        # 1) Save each radar data into a single .npy file and downsampling these data
             python save_meteonet.py
     
        # 2) Using sliding window approach to partition model inputs and ground truth data
             python split_meteonet.py
        ```
   * **Directory layout after processing**:
       ```
       data/meteonet/data/nw/reflectivity_split/
       ├── 20160101_0000.npy
       ├── 20160101_0115.npy
       ├── ......
       └── 20181031_2120.npy
       ```
     
2. **SEVIR-LR dataset**:
   * **Download**:
      Visit [https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip](https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip)
   
   * **Processing**:
      We provide three helper scripts to convert the raw HDF5 file into NumPy arrays and to split out individual precipitation events
        ```
        # 1) Convert the raw .h5 file to .npy array
             python process_sevir.py
        
        # 2) Split each precipitation event into a single .npy file
             python save_sevir.py
     
        # 3) Use sliding window to split each precipitation event into 3 npy files.
             python split_sevir.py
        ```
   * **Directory layout after processing**:
     ```
     data/SEVIR/data/vil_single/
     ├── random/
     └── storm/
     ```

---

## 🏃‍ Quick Start

### Train a model

1. **For SEVIR-LR dataset**:
    ```
    python train_baseline_sevir.py \
      --model afno \
      --batchsize 16 \
      --epoch 100 \
      --lr 1e-3 \
      --gpus 0
    ```
   
2. **For MeteoNet dataset**:
    ```
    python train_meteonet.py \
      --model m_afno \
      --batchsize 16 \
      --epoch 100 \
      --lr 1e-3 \
      --gpus 0
    ```