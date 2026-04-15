# SAR Oil Spill Detection

A U-Net based semantic segmentation system for detecting oil spills in Synthetic Aperture Radar (SAR) satellite images.

## Overview

This project uses deep learning to automatically segment oil spill regions in SAR imagery. SAR sensors are ideal for ocean monitoring as they work through clouds and at night. The model performs pixel-level classification to produce binary masks highlighting oil spill locations.

## Model

- **Architecture**: U-Net (convolutional encoder-decoder with skip connections)
- **Input**: 256×256 grayscale SAR images
- **Output**: 256×256 binary mask (oil spill vs background)
- **Training**: Binary cross-entropy loss, Adam optimizer
- **Performance**: IoU ~0.58, Dice ~0.69

## Dataset

- Deep SAR Oil Spill Segmentation (Refined) from Kaggle
- PALSAR and Sentinel-1 SAR images with expert-annotated masks
- Data augmentation: horizontal/vertical flips, 90° rotations (5000+ training samples)

## Requirements

```
TensorFlow >= 2.x
OpenCV
NumPy
Streamlit
```

Install dependencies:

```bash
pip install tensorflow opencv-python numpy streamlit
```

## Usage

Run the dashboard:

```bash
streamlit run App1.py
```

Upload SAR images or select from validation samples to see predictions. The app displays:
- Input image
- Predicted mask
- Overlay visualization
- IoU/Dice metrics (for validation data)

<img width="1416" height="844" alt="image" src="https://github.com/user-attachments/assets/7447745e-915a-497e-98a6-f7718442471c" />
<img width="1372" height="666" alt="image" src="https://github.com/user-attachments/assets/fecc3b25-f19f-4ec8-80f2-b4166708fdbe" />



## Files

- `App1.py` - Streamlit dashboard
- `unet_oilspill.h5` - Trained model weights
- `oil_spill.ipynb` - Training notebook
- `val_images/` - Validation dataset
- `docs.txt` - Full project explanation
