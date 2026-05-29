import os
import cv2
import numpy as np
import pickle
import time
import argparse
import logging

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from src.utils import load_config, setup_logging

logger = setup_logging()

def run_predictions(image_path, model_dir="models", output_dir="predictions"):
    # 1. Load config
    config = load_config()
    image_size = config['dataset']['image_size']
    
    # 2. Check and load artifacts
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    pca_path = os.path.join(model_dir, "pca.pkl")
    classes_path = os.path.join(model_dir, "class_names.pkl")
    
    cnn_path = os.path.join(model_dir, "best_plant_disease_model_cnn.keras")
    pca_model_path = os.path.join(model_dir, "best_plant_disease_model_pca_dense.keras")

    if not HAS_TENSORFLOW:
        # Check for pickle models instead of .keras in fallback mode
        cnn_path = cnn_path.replace(".keras", ".pkl")
        pca_model_path = pca_model_path.replace(".keras", ".pkl")

    missing = [p for p in [scaler_path, pca_path, classes_path, cnn_path, pca_model_path] if not os.path.exists(p)]
    if len(missing) > 0:
        logger.error(f"Missing required model or preprocessing files: {missing}")
        logger.error("Please run the training pipeline first: python src/train.py")
        return

    # Load artifacts
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    with open(classes_path, "rb") as f:
        class_names = pickle.load(f)

    logger.info(f"Loaded class mapping: {class_names}")

    # Load models
    if not HAS_TENSORFLOW:
        logger.info("TensorFlow not installed. Loading Scikit-learn fallback models...")
        with open(cnn_path, "rb") as f:
            cnn_model = pickle.load(f)
        with open(pca_model_path, "rb") as f:
            pca_dense_model = pickle.load(f)
    else:
        logger.info("Loading TensorFlow/Keras models...")
        cnn_model = load_model(cnn_path)
        pca_dense_model = load_model(pca_model_path)

    # 3. Read image with OpenCV
    if not os.path.exists(image_path):
        logger.error(f"Image not found at path: {image_path}")
        return

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        logger.error(f"Unable to read image at path: {image_path}")
        return

    # Convert BGR to RGB for predictions
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to the resolution the model expects
    img_resized = cv2.resize(img_rgb, (image_size, image_size))

    # 4. Prepare image for CNN
    cnn_input = img_resized.astype(np.float32) / 255.0
    cnn_input = np.expand_dims(cnn_input, axis=0) # shape: (1, H, W, C)

    # 5. Prepare image for PCA + Dense ANN
    flat_input = img_resized.reshape(1, -1).astype(np.float32)
    scaled_input = scaler.transform(flat_input)
    pca_input = pca.transform(scaled_input) # shape: (1, n_components)

    # 6. Run CNN Inference and time it
    t0 = time.time()
    cnn_probs = cnn_model.predict(cnn_input, verbose=0)
    cnn_time = (time.time() - t0) * 1000 # in ms
    cnn_class_idx = np.argmax(cnn_probs[0])
    cnn_confidence = cnn_probs[0][cnn_class_idx]
    cnn_pred_class = class_names[cnn_class_idx]

    # 7. Run PCA + Dense Inference and time it
    t0 = time.time()
    pca_probs = pca_dense_model.predict(pca_input, verbose=0)
    pca_time = (time.time() - t0) * 1000 # in ms
    pca_class_idx = np.argmax(pca_probs[0])
    pca_confidence = pca_probs[0][pca_class_idx]
    pca_pred_class = class_names[pca_class_idx]

    # 8. Output results
    logger.info("=====================================================")
    logger.info("INFERENCE COMPARISON SUMMARY")
    logger.info("=====================================================")
    logger.info(f"Image: {os.path.basename(image_path)}")
    logger.info("-----------------------------------------------------")
    logger.info("1. CONVOLUTIONAL NEURAL NETWORK (CNN) / FALLBACK")
    logger.info(f"   Predicted Class: {cnn_pred_class}")
    logger.info(f"   Confidence Score: {cnn_confidence * 100:.2f}%")
    logger.info(f"   Inference Time: {cnn_time:.2f} ms")
    logger.info("-----------------------------------------------------")
    logger.info("2. PCA + DENSE ARTIFICIAL NEURAL NETWORK (ANN) / FALLBACK")
    logger.info(f"   Predicted Class: {pca_pred_class}")
    logger.info(f"   Confidence Score: {pca_confidence * 100:.2f}%")
    logger.info(f"   Inference Time: {pca_time:.2f} ms")
    logger.info("=====================================================")

    # 9. Create visualization overlay and save
    os.makedirs(output_dir, exist_ok=True)
    vis_img = img_bgr.copy()
    h, w, _ = vis_img.shape
    
    # Scale text size based on image size
    font_scale = max(0.5, w / 800.0)
    thickness = max(1, int(w / 400))
    
    # Write CNN prediction in top-left
    cv2.putText(
        vis_img, 
        f"CNN: {cnn_pred_class} ({cnn_confidence*100:.1f}%)", 
        (10, int(30 * font_scale)), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (46, 204, 113), # Green (BGR: 113, 204, 46)
        thickness, 
        cv2.LINE_AA
    )
    
    # Write PCA prediction below
    cv2.putText(
        vis_img, 
        f"PCA: {pca_pred_class} ({pca_confidence*100:.1f}%)", 
        (10, int(60 * font_scale)), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        (219, 152, 52), # Blue (BGR: 52, 152, 219)
        thickness, 
        cv2.LINE_AA
    )

    out_path = os.path.join(output_dir, f"prediction_output_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, vis_img)
    logger.info(f"Saved prediction overlay to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plant Disease Classification Prediction Utility")
    parser.add_argument("--image", type=str, required=True, help="Path to input plant leaf image")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where trained models are saved")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save prediction visual overlay")
    
    args = parser.parse_args()
    run_predictions(args.image, args.model_dir, args.output_dir)

if __name__ == "__main__":
    main()
