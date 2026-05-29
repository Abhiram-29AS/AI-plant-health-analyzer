import os
import yaml
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_config(config_path="config.yaml"):
    """
    Loads configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging():
    """
    Sets up a professional logging system.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("PlantDiseaseClassification")

def plot_training_curves(cnn_history, pca_history, save_dir):
    """
    Plots training curves (accuracy & loss) for both CNN and PCA + Dense models and saves the figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Loss Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if cnn_history:
        plt.plot(cnn_history.history['loss'], label='CNN Train Loss', color='#2ecc71', linestyle='-', linewidth=2)
        plt.plot(cnn_history.history['val_loss'], label='CNN Val Loss', color='#27ae60', linestyle='--', linewidth=2)
    if pca_history:
        plt.plot(pca_history.history['loss'], label='PCA+Dense Train Loss', color='#3498db', linestyle='-', linewidth=2)
        plt.plot(pca_history.history['val_loss'], label='PCA+Dense Val Loss', color='#2980b9', linestyle='--', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Accuracy Comparison
    plt.subplot(1, 2, 2)
    if cnn_history:
        plt.plot(cnn_history.history['accuracy'], label='CNN Train Acc', color='#2ecc71', linestyle='-', linewidth=2)
        plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Acc', color='#27ae60', linestyle='--', linewidth=2)
    if pca_history:
        plt.plot(pca_history.history['accuracy'], label='PCA+Dense Train Acc', color='#3498db', linestyle='-', linewidth=2)
        plt.plot(pca_history.history['val_accuracy'], label='PCA+Dense Val Acc', color='#2980b9', linestyle='--', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "accuracy_loss_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.getLogger("PlantDiseaseClassification").info(f"Saved accuracy and loss curves comparison to {save_path}")

def plot_confusion_matrix(y_true, y_pred, classes, save_dir, model_name="model"):
    """
    Plots confusion matrix and saves the figure.
    """
    from sklearn.metrics import confusion_matrix
    
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Greens" if "cnn" in model_name.lower() else "Blues", 
        xticklabels=classes, 
        yticklabels=classes
    )
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name.lower()}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.getLogger("PlantDiseaseClassification").info(f"Saved confusion matrix for {model_name} to {save_path}")
