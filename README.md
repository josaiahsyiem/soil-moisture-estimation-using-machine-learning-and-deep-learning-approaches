# Soil Moisture Retrieval from Sentinel-1 Dual-Pol SAR Data using Physics-Inspired Machine Learning

**Author:** Josaiah Syiem  
**Institution:** Centre of Studies in Resources Engineering (CSRE), IIT Bombay  
**Supervisor:** Prof. Y.S. Rao  
**Programme:** M.Tech in Geo-informatics and Natural Resources Engineering

---

## Overview

This project develops a machine learning pipeline to estimate volumetric soil moisture content (SMC) from **Sentinel-1 GRD dual-polarization SAR data** over vegetated terrain. The study uses data from the **Texas Soil Observation Network (TxSON)**, which consists of 38 in-situ monitoring stations across cropland and shrubland in central Texas. After matching Sentinel-1 acquisitions with ground observations, the final dataset contains approximately **3,347 samples** collected between **2015 and 2019**.

One of the main challenges in SAR-based soil moisture estimation is that vegetation attenuates and scatters the radar signal, making it difficult to separate the soil moisture response from the vegetation response. Instead of relying only on raw SAR backscatter, this work uses **physics-inspired feature engineering** based on electromagnetic scattering theory to better capture the relationship between SAR observations and soil moisture.

---

## Key Results

| Metric | Lin Reg | RF | GRNN | SVR | **CNNR + Eng** | CNNR + Raw |
|--------|---------|-----|------|-----|---------------|------------|
| RMSE (m³/m³) | 0.0648 | 0.0769 | 0.0742 | 0.0571 | **0.0552** | 0.0774 |
| MAE (m³/m³) | 0.0533 | 0.0658 | 0.0657 | 0.0455 | **0.0437** | 0.0631 |
| Pearson r | 0.7061 | 0.5700 | 0.5940 | 0.7587 | **0.7728** | 0.4593 |
| R² | 0.4287 | 0.1961 | 0.2510 | 0.5567 | **0.5867** | 0.1859 |

All models were evaluated using **station-grouped cross-validation**, where the test stations are completely excluded from training. This provides a more realistic evaluation of how well the models generalize to unseen locations.

> For comparison, the physics-based benchmark by Bhogapurapu et al. (2022) reports an RMSE of **0.048-0.055 m³/m³** and a **Pearson correlation of 0.79-0.85** on the same dataset.

---

## Methodology

### 1. Physics-Inspired SAR Feature Engineering

Two physics-inspired SAR features were derived following Bhogapurapu et al. (2022).

**DpRVIc (Dual-Polarization Radar Vegetation Index, corrected for GRD):**

$$
\mathrm{DpRVIc} = \frac{q(q+3)}{(q+1)^2},
\qquad
q = \frac{\sigma^0_{VH}}{\sigma^0_{VV}}
\text{ (linear scale)}
$$

This feature estimates vegetation canopy conditions using only SAR observations, without requiring optical imagery. Its values range from **0 (bare soil)** to **1 (dense vegetation)**.

**Δσ (Change Detection Parameter):**

$$
\Delta\sigma = \sigma^0_{VV} - \sigma^0_{dry}
$$

where **σ⁰dry** is calculated as the **2nd percentile of historical VV backscatter** for each station. This serves as a robust estimate of the driest observed condition and helps isolate moisture-driven backscatter changes from static effects such as terrain, soil type, and permanent vegetation.

### 2. Ancillary Soil Properties

Station metadata from TxSON (Caldwell et al., 2019) was used to incorporate additional soil information, including:

- **Clay (%)**
- **Sand (%)**
- **Bulk Density (BD)**, which influences soil porosity and drainage
- **Elevation**, which affects local drainage patterns

### 3. Evaluation Protocol

All models use **GroupShuffleSplit (80/20)** based on station IDs, ensuring that stations in the test set are never seen during training. This prevents the model from learning station-specific patterns and avoids the spatial data leakage that often occurs when random train-test splits are used.

### 4. Models Compared

| Model | Type | Notes |
|-------|------|------|
| Linear Regression | Classical baseline | Full standardized feature set |
| Random Forest | Ensemble learning | Tuned with depth = 16, n_estimators = 100, max_features = 0.7 |
| GRNN | Shallow neural network | Specht (1991); σ optimized using group-aware cross-validation |
| SVR | Kernel-based regression | RBF kernel with C, γ, and ε optimized using group-aware cross-validation |
| CNNR + Engineered Features | 1D CNN | Uses the same engineered features as the Random Forest model |
| CNNR + Raw Features | 1D CNN | Evaluates whether a CNN can learn directly from raw features without physics-inspired feature engineering |

### 5. SAE Data Augmentation (Dabboor et al., 2023)

A **Sparse Autoencoder (SAE)** was implemented for data augmentation following Dabboor et al. (2023), with one important methodological correction. In the original study, the autoencoder was trained on the complete dataset before the train-test split, allowing near-duplicate test samples to influence training through data leakage.

In this work, the autoencoder was trained only on the training data within each split. Under this leakage-safe protocol, SAE augmentation produced no consistent improvement over the original dataset. Its performance was comparable to simple Gaussian noise augmentation, suggesting that the improvements reported in the original study were largely influenced by data leakage.

### 6. Spatial Generalization Analysis

A multi-seed evaluation was carried out across eight different station-based train-test splits. Pearson correlation varied from **0.15 to 0.71**, depending only on which stations were held out for testing. This shows that **spatial generalization remains the biggest challenge**, and that single train-test splits can significantly overestimate real-world performance on unseen locations.

---

## Repository Structure

```text
smc-soil-moisture-retrieval/
├── README.md
├── requirements.txt
├── notebook/
│   └── smc_soil_moisture_retrieval.ipynb   # Main pipeline
├── gee/
│   └── SMC_Sentinel1_Preprocessing.js      # GEE preprocessing script
├── results/
│   └── predicted_vs_measured.png           # Scatter plot output
└── data/
    └── README.md                           # Data access instructions
```

---

## Pipeline

```text
Sentinel-1 GRD imagery
        │
        ▼
Google Earth Engine preprocessing
(DpRVIc, Hc, Θc, mc, VV, VH, incidence angle extracted at 38 TxSON stations)
        │
        ▼
Python ML pipeline (smc_soil_moisture_retrieval.ipynb)
        │
        ├── Feature engineering (DpRVIc, Δσ, sigma_dry, VV_dB, soil texture)
        ├── Station-grouped train/test split (GroupShuffleSplit)
        ├── Feature selection (RFECV + manual curation)
        ├── Model training (LR, RF, SVR, GRNN, CNNR)
        ├── SAE augmentation + noise ablation + robustness check
        └── Evaluation on held-out stations
```
## Data

This study uses the **Texas Soil Observation Network (TxSON)** dataset, which is publicly available at:

> Caldwell, T., Bongiovanni, T. et al. (2019). *Texas Soil Observation Network (TxSON)*. Texas Data Repository, V5.  
> https://doi.org/10.18738/T8/JJ16CF

Station-level soil properties, including **clay content, sand content, bulk density, and elevation**, are obtained from the `metadataTxSON.csv` file distributed with the same dataset.

### To run this notebook

1. Download the TxSON dataset using the DOI above.
2. Place `36535740.xlsx` and `metadataTxSON.csv` inside a `data/` folder.
3. Update the `FILE_PATH` and `META_PATH` variables in Cells 4 and 15.
4. Install the required packages:

```bash
pip install -r requirements.txt
```

5. Run the notebook from top to bottom.

---

## GEE Preprocessing Script

The file `gee/SMC_Sentinel1_Preprocessing.js` contains the complete SAR preprocessing workflow implemented in **Google Earth Engine**. It computes **DpRVIc**, **Hc**, **Theta_c**, **mc**, and extracts the corresponding time series for all TxSON station locations.

Although the TxSON dataset already contains the extracted values used in this study, the script allows the entire preprocessing workflow to be reproduced or applied to a different study area.

To use the script:

1. Open the **Google Earth Engine Code Editor**.
2. Paste the script into a new project.
3. Update the date range and station coordinates if needed.
4. Run the export tasks.

---

## CNNR Architecture

The CNN Regression (CNNR) model follows the architecture proposed by **Liu et al. (2021)** (Figure 2) and is implemented in **PyTorch**.

```text
Input (1 × n_features)
    │
    Conv1D(1→16, k=1) + BatchNorm + ReLU
    │
    ┌──────────────┬──────────────┬──────────────┐
Conv1D(16→32, k=2) Conv1D(16→32, k=3) Conv1D(16→32, k=4)
    └──────────────┴──────────────┴──────────────┘
    │
Concatenate (Inception-style)
    │
Conv1D(32→64, k=3) + BatchNorm + ReLU
    │
Conv1D(64→128, k=3) + BatchNorm + ReLU
    │
Flatten
    │
FC(128) + ReLU + Dropout(0.5)
    │
FC(1)
```

The network does **not** use pooling layers so that all feature information is preserved throughout the model. Training uses the **Adam optimizer** with weight decay, a **ReduceLROnPlateau** learning-rate scheduler, and **early stopping (patience = 100)**.

---

## Key Findings

1. **Physics-inspired feature engineering had the biggest impact on model performance.** Features such as **DpRVIc** and **Δσ** consistently improved soil moisture estimation compared to using raw SAR backscatter alone.

2. **Random Forest initially achieved the best performance** using the engineered SAR features, reaching a **Pearson correlation of approximately 0.71** under station-based cross-validation.

3. **Adding static soil properties** such as **clay content, sand content, and bulk density** improved the performance of simpler models like **Linear Regression**, but reduced the generalization ability of **Random Forest**. After including these variables, **RFECV no longer selected the engineered SAR features as the most important predictors**, suggesting that the static soil variables dominated the feature selection process.

4. **CNNR with engineered features achieved the best overall performance**, with **RMSE = 0.0552 m³/m³**, **MAE = 0.0437 m³/m³**, **Pearson r = 0.7728**, and **R² = 0.5867**. In contrast, CNNR trained on the raw feature set performed considerably worse, showing that the physics-inspired features remained essential even for deep learning.

5. **SAE data augmentation did not provide a consistent improvement.** Under a leakage-safe evaluation protocol, the original dataset, SAE-augmented dataset, and Gaussian noise augmentation produced very similar results. This suggests that the large improvements reported by Dabboor et al. (2023) were largely influenced by data leakage.

6. **Spatial generalization remains the biggest challenge.** Across multiple station-based train-test splits, Pearson correlation varied from **0.15 to 0.71**, showing that model performance depends strongly on which stations are held out for testing. This highlights the importance of evaluating models using multiple spatial splits rather than relying on a single train-test partition.

---

## References

- Bhogapurapu, N., Dey, S., Homayouni, S., Bhattacharya, A., & Rao, Y.S. (2022). *Field-scale soil moisture estimation using Sentinel-1 GRD SAR data*. **Advances in Space Research**, 70(10), 3845-3858. https://doi.org/10.1016/j.asr.2022.03.019

- Dabboor, M., Atteia, G., Meshoul, S., & Alayed, W. (2023). *Deep Learning-Based Framework for Soil Moisture Content Retrieval of Bare Soil from Satellite Data*. **Remote Sensing**, 15(7), 1916. https://doi.org/10.3390/rs15071916

- Liu, J., Xu, Y., Li, H., & Guo, J. (2021). *Soil Moisture Retrieval in Farmland Areas with Sentinel Multi-Source Data Based on Regression Convolutional Neural Networks*. **Sensors**, 21(3), 877. https://doi.org/10.3390/s21030877

- Caldwell, T., Bongiovanni, T. et al. (2019). *Texas Soil Observation Network (TxSON)*. Texas Data Repository, V5. https://doi.org/10.18738/T8/JJ16CF

- Dabboor, M., Xu, O.J., Vakalopoulou, M. et al. (2024). *The RADARSAT Constellation Mission for Soil Moisture Retrieval of Bare Soil by Compact Polarimetry and Random Forest Regression*. **Canadian Journal of Remote Sensing**, 50(1), 2356688.

- Santi, E., Dabboor, M., Pettinato, S., & Paloscia, S. (2019). *Combining Machine Learning and Compact Polarimetry for Estimating Soil Moisture from C-Band SAR Data*. **Remote Sensing**, 11(20), 2451.

- Chung, J., Lee, Y., Kim, J., Jung, C., & Kim, S. (2022). *Soil Moisture Content Estimation Based on Sentinel-1 SAR Imagery Using an Artificial Neural Network and Hydrological Components*. **Remote Sensing**, 14(3), 465.

---

## Tech Stack

`Python` · `scikit-learn` · `PyTorch` · `pandas` · `numpy` · `matplotlib` · `scipy` · `Google Earth Engine (JavaScript API)`
