# Plant Disease Classification Model

A deep learning-based plant disease classification system built using TensorFlow/Keras and the PlantVillage dataset. The project classifies plant diseases from leaf images and provides predictions for unseen samples through a simple inference pipeline.

## Project Overview

This project implements an end-to-end machine learning workflow for plant disease detection using image classification techniques.

Key features include:

* Image preprocessing and normalization
* CNN-based disease classification
* PCA-based feature reduction pipeline
* StandardScaler normalization
* Model evaluation using confusion matrix and classification reports
* Training history visualization
* Inference on custom leaf images

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn

## Dataset

The project uses the PlantVillage dataset containing healthy and diseased leaf images across multiple plant species.

Expected dataset structure:

Dataset/
└── PlantVillage/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
├── class_1/
├── class_2/
└── ...

## Machine Learning Pipeline

1. Load and preprocess leaf images
2. Resize images to model input dimensions
3. Normalize image data
4. Apply StandardScaler
5. Apply PCA for dimensionality reduction
6. Train CNN-based classifier
7. Evaluate model performance
8. Generate predictions for new images

## Project Structure

project_plant_dc/

├── config.yaml
├── requirements.txt
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── Dataset/
├── models/
├── reports/
└── predictions/

## Requirements

* Python 3.13
* tensorflow-cpu==2.21.0
* Dependencies listed in requirements.txt

## Installation

Create a virtual environment:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## Training

Train the models from scratch:

```powershell
py -3.13 -m src.train
```

Expected outputs:

* best_plant_disease_model_cnn.keras
* best_plant_disease_model_pca_dense.keras
* scaler.pkl
* pca.pkl
* class_names.pkl
* confusion matrices
* classification report
* training history plots

## Running Inference

Predict disease from a leaf image:

```powershell
py -3.13 -m src.predict --image "C:\path\to\leaf.jpg"
```

Prediction visualizations are saved in:

predictions/

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report
* Training/Validation Loss Curves
* Training/Validation Accuracy Curves

## Results

* Achieved approximately 89% classification accuracy on the validation dataset.
* Successfully distinguishes multiple plant disease categories from leaf images.

## Future Improvements

* Deploy as a web application
* Add real-time camera inference
* Expand support for additional crop species
* Improve performance using transfer learning
