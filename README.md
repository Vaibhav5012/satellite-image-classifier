# ResNet EuroSAT Satellite Image Classification

A complete PyTorch implementation for training ResNet models on the EuroSAT dataset for satellite image classification. This project demonstrates transfer learning techniques for land use and land cover (LULC) classification from Sentinel-2 satellite imagery.

## Features

- **Transfer Learning**: Pretrained ResNet models (ResNet-18 to ResNet-152) fine-tuned on EuroSAT dataset
- **Data Augmentation**: Comprehensive augmentation techniques including rotation, flipping, and color jittering
- **Complete Pipeline**: End-to-end training, validation, testing, and inference
- **Visualization**: Training history plots, confusion matrices, and prediction visualizations
- **Google Colab Ready**: Fully compatible with Google Colab for cloud-based training
- **GPU Support**: Automatic CUDA detection and GPU utilization

## Dataset

The EuroSAT dataset contains 27,000 labeled Sentinel-2 satellite images (64×64 pixels) across 10 land use classes:
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

## Requirements

- Python 3.7+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- tqdm

## Quick Start

1. Clone this repository
2. Install dependencies: `pip install torch torchvision matplotlib scikit-learn tqdm`
3. Run the training script in Google Colab or locally
4. Monitor training progress with real-time visualizations
5. Evaluate performance on the test set with confusion matrix and classification metrics

## Model Performance

- **Architecture**: ResNet-50 with ImageNet pretrained weights
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam with learning rate scheduling
- **Expected Test Accuracy**: 85-95% depending on hyperparameters

## Results

The trained model generates:
- Training/validation loss and accuracy curves
- Confusion matrix on test set
- Classification report with precision, recall, and F1-scores
- Prediction visualizations with confidence scores

## Usage

```python
# Predict on a single image
predicted_class, confidence = predict_single_image(
    model, 
    'path/to/image.jpg', 
    class_names
)
print(f"Prediction: {predicted_class} ({confidence:.2f}%)")
```

## References

- EuroSAT Dataset: http://madm.dfki.de/sentinel/
- ResNet Paper: He et al., 2015
- PyTorch Documentation: https://pytorch.org

## License

MIT License - feel free to use for research and educational purposes.
