# Vision-Based GeoFM for EuroSAT Land Cover Classification

## Overview
This project presents a Vision-Based Geographic Foundation Model (GeoFM) adapted for Land Use and Land Cover (LULC) classification using multispectral satellite imagery. By incorporating a custom channel adapter (Conv 1x1) with a pre-trained Vision Transformer (ViT), we successfully bridge the gap between 13-band Sentinel-2 multispectral data and models trained on standard 3-channel RGB imagery. The system achieves a state-of-the-art classification performance of over 98.5% accuracy across 10 diverse land cover classes.

## Dataset
We utilize the **EuroSAT-MS** (Multispectral) dataset, which consists of:
*   **Total Images**: 27,000 satellite patches.
*   **Resolution**: 64x64 pixels per patch.
*   **Bands**: 13 spectral bands from the Sentinel-2 satellite (including Visible, Near-Infrared, and Shortwave Infrared bands).
*   **Classes**: 10 land cover categories:
    1.  Annual Crop (*Tanaman Semusim*)
    2.  Forest (*Hutan*)
    3.  Herbaceous Vegetation (*Vegetasi Herba*)
    4.  Highway (*Jalan Raya*)
    5.  Industrial (*Kawasan Industri*)
    6.  Pasture (*Padang Rumput*)
    7.  Permanent Crop (*Perkebunan Tetap*)
    8.  Residential (*Pemukiman*)
    9.  River (*Sungai*)
    10. Sea & Lake (*Laut dan Danau*)

## Methodology
To preserve the powerful features of a pre-trained Vision Transformer (ViT) while adapting it to multispectral inputs, we implemented the following pipeline:

1.  **Preprocessing & Feature Engineering**:
    *   **Loading**: Custom loading using `tifffile` to handle raw 13-band `.tif` data.
    *   **Normalization**: Z-score normalization using channel-specific statistics calculated from a representative subset of the dataset.
    *   **Splitting**: Stratified split preserving class balances with a 70% Train, 15% Validation, and 15% Test ratio.
    *   **Data Augmentations**: Top-down compatible spatial augmentations (Random Horizontal & Vertical Flips, 90-degree rotations) and resizing to 224x224 to match ViT positional embeddings.

2.  **Model Architecture (`GeoFM_ViT`)**:
    *   **Channel Adapter**: A custom `nn.Conv2d(13, 3, kernel_size=1)` layer coupled with a `nn.BatchNorm2d(3)`. This projects the 13 multispectral bands down to 3 compact, information-dense channels without losing pixel-aligned geographic spatial information.
    *   **Backbone**: Pre-trained Vision Transformer (`google/vit-base-patch16-224-in21k`) initialized with ImageNet-21k weights.
    *   **Classifier Head**: A linear classification layer mapping the ViT output `[CLS]` token to the 10 target classes.

## Results
The model was fine-tuned for 5 epochs using the AdamW optimizer with Cosine Annealing Learning Rate scheduling and Automatic Mixed Precision (AMP) on a Tesla T4 GPU. The final evaluations on the independent **Test Set** yielded outstanding metrics:

*   **Test Accuracy**: 98.54%
*   **Test F1-score (Macro)**: 0.9850
*   **Test F1-score (Weighted)**: 0.9854

### Confusion Matrix Insights
The confusion matrix highlights near-flawless classification. Minor prediction errors were exclusively confined to subtle transitional agricultural classes (e.g., mistaking small portions of *Annual Crop* for *Permanent Crop* or *Pasture*), which are highly similar even to human annotators.

## Tech Stack
*   **Language**: Python 3
*   **Deep Learning Framework**: PyTorch, Torchvision
*   **Transformers Library**: Hugging Face Transformers (`transformers`)
*   **Data & Geospatial Loading**: `tifffile`, `numpy`, `pandas`
*   **Visualization**: `matplotlib`, `seaborn`
*   **Metrics**: `scikit-learn`

