#  Plant Disease Classification Model

A deep learning-based plant disease classification system developed using TensorFlow/Keras and the PlantVillage dataset. The project identifies plant diseases from leaf images and provides predictions on unseen samples using trained classification models.

---

##  Project Overview

Plant diseases can significantly impact agricultural productivity. This project leverages computer vision and deep learning techniques to automatically classify plant diseases from leaf images.

The pipeline includes:

* Image preprocessing and normalization
* Convolutional Neural Network (CNN) training
* PCA-based dimensionality reduction
* StandardScaler normalization
* Model evaluation using confusion matrices and classification reports
* Visualization of training performance
* Prediction on custom leaf images

---

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

##  Dataset

This project uses the **PlantVillage Dataset** for plant disease classification.

Due to GitHub storage limitations, the dataset is **not included** in this repository.

Download the dataset from Kaggle and place it inside the project directory:

```text
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
```

The training pipeline expects the dataset to be available in the above structure.

---

## ⚙️ Machine Learning Pipeline

1. Load and preprocess leaf images
2. Resize and normalize image data
3. Apply StandardScaler
4. Perform PCA-based dimensionality reduction
5. Train CNN and PCA-based models
6. Evaluate performance using classification metrics
7. Generate predictions on unseen images

---

##  Project Structure

```text
classifier/
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── download_dataset.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── reports/
│   ├── accuracy_loss_comparison.png
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_pca_dense.png
│   └── classification_report.txt
│
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd classifier
```

### 2. Create a Virtual Environment

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

##  Training the Model

Before training, ensure the PlantVillage dataset is placed inside the `Dataset/` directory.

Run:

```powershell
py -3.13 -m src.train
```

### Generated Outputs

Training generates:

```text
models/
├── best_plant_disease_model_cnn.keras
├── best_plant_disease_model_pca_dense.keras
├── scaler.pkl
├── pca.pkl
└── class_names.pkl
```

and

```text
reports/
├── accuracy_loss_comparison.png
├── confusion_matrix_cnn.png
├── confusion_matrix_pca_dense.png
└── classification_report.txt
```

---

##  Running Inference

Predict disease from a leaf image:

```powershell
py -3.13 -m src.predict --image "path_to_leaf_image.jpg"
```

Prediction outputs are saved inside:

```text
predictions/
```

---

##  Evaluation Metrics

The models are evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report
* Training Loss Curves
* Validation Loss Curves
* Training Accuracy Curves
* Validation Accuracy Curves

---

##  Results

The project successfully classifies plant diseases using deep learning and feature-engineering approaches.

Generated reports include:

* CNN Confusion Matrix
* PCA Dense Model Confusion Matrix
* Accuracy/Loss Comparison Graphs
* Detailed Classification Report

---

##  Future Improvements

* Web-based deployment
* Mobile application integration
* Real-time camera inference
* Transfer learning with advanced architectures
* Support for additional crop species

**Abhiramjee Pittu**

B.Tech – Artificial Intelligence & Data Science

Indian Institute of Information Technology, Sri City
