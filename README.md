# Satellite Image Classifier

https://github.com/user-attachments/assets/6e51dd2a-bb90-48c2-b6bf-d792a2b1d2f0

A complete solution for land use and land cover classification from Sentinel-2 satellite images using deep learning (ResNet-50, EuroSAT). Includes both a Streamlit web app and a Python desktop GUI app.

---

## 🚀 Project Overview

This project automatically classifies satellite images into 10 standard EuroSAT land use categories (forest, urban, crops, water, etc.) using a state-of-the-art convolutional neural network. Designed for researchers, students, and environmental analysts.

---

## ✨ Features

- **Upload Satellite Images** and receive instant predictions
- **Probability Breakdown:** See prediction confidence for all 10 classes
- **Visualization:** Bar charts of class probabilities
- **Batch Processing** (Streamlit): Classify multiple images at once, CSV download
- **Streamlit Web App**: Modern browser interface, works in Colab
- **GPU Support**: Uses CUDA/GPU if available

---

## 🔧 Installation & Requirements

**Dependencies** (Python 3.7+ recommended):
```
torch
torchvision
pillow
numpy
matplotlib
scikit-learn
tqdm
streamlit      # For Streamlit app
```

**Install all dependencies:**
```
pip install torch torchvision pillow numpy matplotlib scikit-learn tqdm streamlit
```

---

## ⚡ Quick Start

### Streamlit Web App

```
streamlit run app.py
```
Or in Google Colab:
```
!pip install streamlit torch torchvision pillow numpy matplotlib scikit-learn tqdm -q
!streamlit run app.py --server.headless true
```

### Python GUI Desktop App

```
python satellite_classifier.py
```

---

## 🏷️ EuroSAT Land Use Classes

- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea Lake

---

## 📝 Contribution

Pull requests and suggestions are welcome!  
Please open Issues for feature requests or bug reports.

---

## 📄 License

MIT License - Free for academic and commercial use.

---

## 📚 References

- EuroSAT: http://madm.dfki.de/sentinel/
- ResNet: He et al. (2015)
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/

---

