# Soil Moisture Retrieval from Sentinel-1 SAR Data using Physics-Inspired Machine Learning

**Author:** Josaiah Syiem  
**Institution:** Centre of Studies in Resources Engineering (CSRE), IIT Bombay  
**Supervisor:** Prof. Y.S. Rao  
**Programme:** M.Tech in Geo-informatics and Natural Resources Engineering  

---

## Overview

This project implements an end-to-end machine learning pipeline for estimating
volumetric soil moisture content (SMC) from **Sentinel-1 GRD dual-polarization SAR
data** over vegetated terrain. The study area is the **Texas Soil Observation Network
(TxSON)** — 38 in-situ stations across cropland and shrubland in central Texas —
with ~3,347 matched SAR/ground-truth observations spanning 2015–2019.

The central challenge is that vegetation attenuates and scatters the radar signal,
confounding the soil moisture response. This study addresses that challenge through
**physics-inspired feature engineering** derived from electromagnetic scattering
theory, rather than relying on raw backscatter alone.

---

## Key Results

| Metric | Lin Reg | RF | GRNN | SVR | **CNNR+eng** | CNNR+raw |
|--------|---------|-----|------|-----|----------|----------|
| RMSE (m³/m³) | 0.0648 | 0.0769 | 0.0742 | 0.0571 | **0.0552** | 0.0774 |
| MAE (m³/m³) | 0.0533 | 0.0658 | 0.0657 | 0.0455 | **0.0437** | 0.0631 |
| Pearson r | 0.7061 | 0.5700 | 0.5940 | 0.7587 | **0.7728** | 0.4593 |
| R² | 0.4287 | 0.1961 | 0.2510 | 0.5567 | **0.5867** | 0.1859 |

All models evaluated under **station-grouped cross-validation** — test stations are
entirely absent from training, providing a realistic assessment of spatial
generalization to unseen locations.

> For comparison, the physics-based benchmark on this dataset (Bhogapurapu et al.,
> 2022) reports RMSE 0.048–0.055 m³/m³ and Pearson r = 0.79–0.85.

---

## Methodology

### 1. Physics-Inspired SAR Feature Engineering

Two features are derived following Bhogapurapu et al. (2022):

**DpRVIc — Dual-pol Radar Vegetation Index (corrected for GRD):**

$$\mathrm{DpRVIc} = \frac{q(q+3)}{(q+1)^2}, \quad q = \frac{\sigma^0_{VH}}{\sigma^0_{VV}} \text{ (linear scale)}$$

Encodes vegetation canopy state from SAR alone (no optical data required).
Ranges from 0 (bare soil) to 1 (dense vegetation).

**Δσ — Change detection parameter:**

$$\Delta\sigma = \sigma^0_{VV} - \sigma^0_{dry}$$

where σ⁰_dry is the per-station 2nd percentile of historical VV backscatter — a
robust proxy for the driest observed condition. Isolates the moisture-driven
backscatter change from the static background (terrain, soil type, permanent
vegetation structure).

### 2. Ancillary Soil Properties

Field-measured soil texture from TxSON station metadata (Caldwell et al., 2019):
- **Clay %** and **Sand %** at 0–10 cm depth
- **Bulk Density (BD)** — soil compaction affects porosity and drainage
- **Elevation** — topographic position influences drainage patterns

### 3. Evaluation Protocol

All models use `GroupShuffleSplit` by station (80/20) so that test stations are
completely held out from training. This prevents the model from memorizing
station-specific moisture baselines — a form of spatial data leakage common in
the literature when random splits are used.

### 4. Models Compared

| Model | Type | Notes |
|---|---|---|
| Linear Regression | Classical baseline | Full feature set, standardized |
| Random Forest | Ensemble ML | Tuned: depth=16, n=100, features=0.7 |
| GRNN | Shallow neural network | Specht (1991); σ tuned via group-aware CV |
| SVR | Kernel machine | RBF kernel; C, γ, ε tuned via group-aware CV |
| CNNR + engineered features | 1D deep CNN | Liu et al. (2021); same features as RF |
| CNNR + raw features | 1D deep CNN | Tests whether CNN can replace feature engineering |

### 5. SAE Data Augmentation (Dabboor et al., 2023)

A **Sparse Autoencoder** was implemented for data augmentation following Dabboor
et al. (2023), with a critical methodological correction: the original study trained
the autoencoder on the full dataset before splitting, allowing near-duplicates of
test samples to enter training (data leakage). Under the corrected leakage-safe
protocol, augmentation produced no reliable improvement across station partitions —
and was matched by plain Gaussian noise — indicating the reported gains in the
original paper were artefactual.

### 6. Spatial Generalization Analysis

Multi-seed robustness evaluation across 8 station partitions shows Pearson r
ranging from 0.15 to 0.71 depending solely on which stations are held out.
**Spatial generalization — not model capacity — is the binding constraint.**
Single-split evaluations substantially overestimate expected performance on truly
unseen locations.

---

## Repository Structure

```
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

```
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

---

## Data

The dataset used in this study is the **TxSON (Texas Soil Observation Network)**
dataset, publicly available at:

> Caldwell, T., Bongiovanni, T. et al. (2019). *Texas Soil Observation Network
> (TxSON)*. Texas Data Repository, V5.
> https://doi.org/10.18738/T8/JJ16CF

Station-level soil properties (clay %, sand %, bulk density, elevation) are from
the network metadata file (`metadataTxSON.csv`) distributed with the same dataset.

**To run this notebook:**
1. Download the TxSON dataset from the DOI above
2. Place `36535740.xlsx` and `metadataTxSON.csv` in a `data/` folder
3. Update `FILE_PATH` and `META_PATH` in Cells 4 and 15 to point to your paths
4. Install dependencies: `pip install -r requirements.txt`
5. Run all cells top to bottom

---

## GEE Preprocessing Script

`gee/SMC_Sentinel1_Preprocessing.js` documents the full SAR preprocessing pipeline
in Google Earth Engine — computing DpRVIc, Hc, Theta_c, mc and extracting time
series at TxSON station locations. The TxSON dataset already contains pre-extracted
values; this script enables reproduction or application to new study areas.

To use: paste into the [GEE Code Editor](https://code.earthengine.google.com/),
update the date range and station coordinates as needed, and run the export tasks.

---

## CNNR Architecture

Following Liu et al. (2021), Figure 2, implemented in PyTorch:

```
Input (1 × n_features)
    │
    Conv1D(1→16, k=1) + BN + ReLU          # depth expansion
    │
    ┌──────────────┬──────────────┐
Conv1D(16→32, k=2) Conv1D(16→32, k=3) Conv1D(16→32, k=4)   # multi-scale
    └──────────────┴──────────────┘
    Concatenate (Inception-style)
    │
    Conv1D(32→64, k=3) + BN + ReLU         # deep extraction
    │
    Conv1D(64→128, k=3) + BN + ReLU
    │
    Flatten → FC(128) + ReLU + Dropout(0.5) → FC(1)
```

No pooling layers (preserves feature information). Training: Adam with weight decay,
ReduceLROnPlateau scheduler, early stopping (patience=100).

---

## Key Findings

1. **Physics-inspired feature engineering was the decisive factor.** DpRVIc and Δσ
   transformed model performance from failure under honest evaluation to competitive
   results. No architecture change alone achieved this.

2. **CNNR with engineered features achieved the best overall performance**
   (r = 0.77, RMSE = 0.055 m³/m³), outperforming all traditional ML models.
   However, CNNR on raw features (r = 0.46) performed worse than Linear Regression
   — confirming the CNN cannot rediscover what physics engineering already encodes
   explicitly, at this dataset size.

3. **Static soil properties (clay %, bulk density) improve simpler models but
   degrade Random Forest generalization.** These features carry strong inter-station
   signal that encourages station-specific overfitting in high-capacity ensembles.

4. **SAE augmentation provides no reliable benefit at this scale.** Under a
   leakage-safe protocol, augmented and non-augmented configurations are
   statistically indistinguishable across partitions. The dramatic gains in Dabboor
   et al. (2023) are attributable to data leakage in their evaluation design.

5. **Spatial generalization is the binding constraint.** Performance varies from
   r = 0.15 to r = 0.71 across partitions — far exceeding any model effect.
   Single-split results should always be contextualized with this variance.

---

## References

- Bhogapurapu, N., Dey, S., Homayouni, S., Bhattacharya, A., & Rao, Y.S. (2022).
  Field-scale soil moisture estimation using Sentinel-1 GRD SAR data. *Advances in
  Space Research*, 70(10), 3845–3858. https://doi.org/10.1016/j.asr.2022.03.019

- Dabboor, M., Atteia, G., Meshoul, S., & Alayed, W. (2023). Deep Learning-Based
  Framework for Soil Moisture Content Retrieval of Bare Soil from Satellite Data.
  *Remote Sensing*, 15(7), 1916. https://doi.org/10.3390/rs15071916

- Liu, J., Xu, Y., Li, H., & Guo, J. (2021). Soil Moisture Retrieval in Farmland
  Areas with Sentinel Multi-Source Data Based on Regression Convolutional Neural
  Networks. *Sensors*, 21(3), 877. https://doi.org/10.3390/s21030877

- Caldwell, T., Bongiovanni, T. et al. (2019). Texas Soil Observation Network
  (TxSON). Texas Data Repository, V5. https://doi.org/10.18738/T8/JJ16CF

- Dabboor, M., Xu, O.J., Vakalopoulou, M. et al. (2024). The RADARSAT Constellation
  Mission for Soil Moisture Retrieval of Bare Soil by Compact Polarimetry and Random
  Forest Regression. *Canadian Journal of Remote Sensing*, 50(1), 2356688.

- Santi, E., Dabboor, M., Pettinato, S., & Paloscia, S. (2019). Combining Machine
  Learning and Compact Polarimetry for Estimating Soil Moisture from C-Band SAR Data.
  *Remote Sensing*, 11(20), 2451.

- Chung, J., Lee, Y., Kim, J., Jung, C., & Kim, S. (2022). Soil Moisture Content
  Estimation Based on Sentinel-1 SAR Imagery Using an Artificial Neural Network and
  Hydrological Components. *Remote Sensing*, 14(3), 465.

---

## Tech Stack

`Python` · `scikit-learn` · `PyTorch` · `pandas` · `numpy` · `matplotlib` ·
`seaborn` · `scipy` · `Google Earth Engine (JavaScript API)`
