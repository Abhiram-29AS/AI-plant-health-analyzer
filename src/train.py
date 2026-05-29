import os
import pickle
import numpy as np
import logging
from sklearn.metrics import classification_report

from src.utils import load_config, setup_logging, plot_training_curves, plot_confusion_matrix
from src.download_dataset import setup_dataset
from src.data import prepare_data_pipelines
from src.model import build_cnn_model, build_pca_dense_model

logger = setup_logging()

def main():
    logger.info("Initializing Plant Disease Classification Training Pipeline...")
    
    # 1. Load configuration
    try:
        config = load_config()
        logger.info("Configuration successfully loaded.")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # 2. Check/Download Dataset
    dataset_path = config['dataset']['path']
    dataset_ready = setup_dataset(dataset_path)
    
    if not dataset_ready:
        logger.error("Dataset pipeline check failed. Please refer to the manual steps above.")
        return

    # 3. Load & Process Data
    try:
        cnn_data, pca_data, y_test_raw, scaler, pca, class_names = prepare_data_pipelines(config)
    except Exception as e:
        logger.error(f"Error during data loading & preprocessing: {e}")
        return

    # Ensure output directories exist
    save_dir = config['training']['save_dir']
    reports_dir = config['training']['reports_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 4. Save preprocessing artifacts
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    pca_path = os.path.join(save_dir, "pca.pkl")
    classes_path = os.path.join(save_dir, "class_names.pkl")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    with open(classes_path, "wb") as f:
        pickle.dump(class_names, f)
    logger.info("Successfully saved scaler.pkl, pca.pkl, and class_names.pkl to models/")

    num_classes = len(class_names)
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']

    # ---------------------------------------------
    # TRAINING CNN MODEL
    # ---------------------------------------------
    logger.info("Building and compiling Convolutional Neural Network (CNN)...")
    X_train_cnn, y_train_oh = cnn_data['train']
    X_val_cnn, y_val_oh = cnn_data['val']
    X_test_cnn, y_test_oh = cnn_data['test']
    
    cnn_model = build_cnn_model(input_shape=X_train_cnn.shape[1:], num_classes=num_classes)
    cnn_model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    cnn_model.summary(print_fn=logger.info)

    logger.info("Training CNN Model...")
    cnn_history = cnn_model.fit(
        X_train_cnn, y_train_oh,
        validation_data=(X_val_cnn, y_val_oh),
        epochs=epochs,
        batch_size=config['dataset']['batch_size'],
        verbose=1
    )

    cnn_save_path = os.path.join(save_dir, "best_plant_disease_model_cnn.keras")
    cnn_model.save(cnn_save_path)
    logger.info(f"CNN model successfully saved to {cnn_save_path}")

    # ---------------------------------------------
    # TRAINING PCA + DENSE MODEL
    # ---------------------------------------------
    logger.info("Building and compiling PCA + Dense Artificial Neural Network (ANN)...")
    X_train_pca, _ = pca_data['train']
    X_val_pca, _ = pca_data['val']
    X_test_pca, _ = pca_data['test']
    
    pca_model = build_pca_dense_model(input_shape=(X_train_pca.shape[1],), num_classes=num_classes)
    pca_model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    pca_model.summary(print_fn=logger.info)

    logger.info("Training PCA + Dense Model...")
    pca_history = pca_model.fit(
        X_train_pca, y_train_oh,
        validation_data=(X_val_pca, y_val_oh),
        epochs=epochs,
        batch_size=config['dataset']['batch_size'],
        verbose=1
    )

    pca_save_path = os.path.join(save_dir, "best_plant_disease_model_pca_dense.keras")
    pca_model.save(pca_save_path)
    logger.info(f"PCA-Dense model successfully saved to {pca_save_path}")

    # ---------------------------------------------
    # VISUALIZATION & REPORTS GENERATION
    # ---------------------------------------------
    logger.info("Generating evaluation plots and reports...")
    
    # 1. Training Curves
    plot_training_curves(cnn_history, pca_history, reports_dir)

    # 2. Evaluation on Test Sets
    logger.info("Evaluating both models on test split...")
    
    # CNN predictions
    cnn_probs = cnn_model.predict(X_test_cnn)
    cnn_preds = np.argmax(cnn_probs, axis=1)
    
    # PCA-Dense predictions
    pca_probs = pca_model.predict(X_test_pca)
    pca_preds = np.argmax(pca_probs, axis=1)

    # Plot Confusion Matrices
    plot_confusion_matrix(y_test_raw, cnn_preds, class_names, reports_dir, model_name="CNN")
    plot_confusion_matrix(y_test_raw, pca_preds, class_names, reports_dir, model_name="PCA_Dense")

    # Generate Classification Reports
    cnn_report = classification_report(y_test_raw, cnn_preds, target_names=class_names)
    pca_report = classification_report(y_test_raw, pca_preds, target_names=class_names)

    report_content = f"""========================================================================
EVALUATION REPORT: PLANT DISEASE CLASSIFICATION MODELS
========================================================================

------------------------------------------------------------------------
1. CONVOLUTIONAL NEURAL NETWORK (CNN) REPORT
------------------------------------------------------------------------
{cnn_report}

------------------------------------------------------------------------
2. PCA + DENSE ARTIFICIAL NEURAL NETWORK (ANN) REPORT
------------------------------------------------------------------------
{pca_report}

========================================================================
"""
    
    report_file_path = os.path.join(reports_dir, "classification_report.txt")
    with open(report_file_path, "w") as f:
        f.write(report_content)
        
    logger.info(f"Classification reports successfully written to {report_file_path}")
    print(report_content)
    logger.info("Training pipeline complete! You can run predictions using src/predict.py.")

if __name__ == "__main__":
    main()
