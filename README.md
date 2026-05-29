# Plant Disease Classification

This project trains and runs inference for a plant disease classification model using TensorFlow and a custom dataset layout.

## Requirements
- Python 3.13
- `tensorflow-cpu==2.21.0`
- Other dependencies listed in `requirements.txt`

## Getting Started

1. Install Python 3.13
   ```powershell
   py -3.13 --version
   ```

2. Copy the full project folder to the new device:
   - `config.yaml`
   - `requirements.txt`
   - `src\`
   - `models\` (if you want to skip training)
   - `Dataset\` (if you want to train from scratch)

3. Create and activate a virtual environment
   ```powershell
   cd "C:\path\to\project_plant dc"
   py -3.13 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install dependencies
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Training the Model

To train the models from scratch, run:
```powershell
py -3.13 -m src.train
```

> Important: use module execution (`-m src.train`). Running `py -3.13 src\train.py` from the project root may fail because imports depend on the `src` package path.

### Expected output files after training
- `models\best_plant_disease_model_cnn.keras`
- `models\best_plant_disease_model_pca_dense.keras`
- `models\scaler.pkl`
- `models\pca.pkl`
- `models\class_names.pkl`
- `reports\accuracy_loss_comparison.png`
- `reports\confusion_matrix_cnn.png`
- `reports\confusion_matrix_pca_dense.png`
- `reports\classification_report.txt`

## Running Inference

To run prediction on an input leaf image:
```powershell
py -3.13 -m src.predict --image "C:\path\to\leaf.jpg"
```

The script will load the trained models from `models\` and save a visualization overlay to:
- `predictions\prediction_output_<filename>.JPG`

## Inference-only setup

If you only need to run predictions, copy:
- `config.yaml`
- `requirements.txt`
- `src\`
- `models\`

Then install dependencies and run the prediction command above.

## Notes
- This project is tested with Python 3.13 and `tensorflow-cpu==2.21.0`.
- The dataset lives in `Dataset\PlantVillage\train` and `Dataset\PlantVillage\val`.
- For inference, any valid image path can be used.
