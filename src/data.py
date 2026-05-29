import os
import cv2
import numpy as np
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def to_categorical(y, num_classes):
    """
    Converts a class vector (integers) to binary class matrix (one-hot encoding).
    """
    return np.eye(num_classes)[y.astype(int)]

logger = logging.getLogger("PlantDiseaseClassification")

def load_and_preprocess_images(dataset_path, image_size):
    """
    Recursively walks the dataset path, reads images with OpenCV, 
    resizes them, and returns image arrays, labels, and class names.
    Supports flat class folders, nested train/val subdirectories, etc.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' does not exist.")
        
    images = []
    labels = []
    class_to_idx = {}
    
    logger.info(f"Scanning directory: {dataset_path} ...")
    
    # Walk directories recursively to collect images
    for root, dirs, files in os.walk(dataset_path):
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if len(img_files) == 0:
            continue
            
        # Limit to 100 images per class subdirectory to prevent MemoryError and optimize training speed
        img_files = img_files[:100]
            
        # Class name is the leaf directory name
        class_name = os.path.basename(root)
        
        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(class_to_idx)
            
        label_idx = class_to_idx[class_name]
        logger.info(f"Found {len(img_files)} images in class directory '{class_name}' ({root})")
        
        for img_name in img_files:
            img_path = os.path.join(root, img_name)
            try:
                # Load image with OpenCV
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize
                img = cv2.resize(img, (image_size, image_size))
                
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")
                continue
                
    if len(images) == 0:
        raise ValueError(f"No valid images found inside dataset directory '{dataset_path}'.")
        
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    
    logger.info(f"Loaded a total of {len(images)} valid images belonging to {len(class_names)} classes.")
    return images, labels, class_names

def prepare_data_pipelines(config):
    """
    Loads, splits, and prepares data for both the Keras CNN and PCA + Dense ANN pipelines.
    """
    dataset_path = config['dataset']['path']
    image_size = config['dataset']['image_size']
    train_split = config['dataset']['train_split']
    val_split = config['dataset']['val_split']
    test_split = config['dataset']['test_split']
    random_seed = config['dataset']['random_seed']
    n_components = config['pca']['n_components']
    
    # 1. Load images and labels
    images, labels, class_names = load_and_preprocess_images(dataset_path, image_size)
    num_classes = len(class_names)
    
    # 2. Stratified train/test/val split
    test_size = test_split / (train_split + val_split + test_split)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, 
        test_size=test_size, 
        random_state=random_seed, 
        stratify=labels
    )
    
    val_size = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size, 
        random_state=random_seed, 
        stratify=y_train_val
    )
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # ---------------------------------------------
    # CNN PIPELINE PREPROCESSING
    # ---------------------------------------------
    X_train_cnn = X_train.astype(np.float32) / 255.0
    X_val_cnn = X_val.astype(np.float32) / 255.0
    X_test_cnn = X_test.astype(np.float32) / 255.0
    
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)
    
    # ---------------------------------------------
    # PCA PIPELINE PREPROCESSING
    # ---------------------------------------------
    logger.info("Initializing PCA dimensionality reduction pipeline...")
    
    num_train, H, W, C = X_train.shape
    X_train_flat = X_train.reshape(num_train, -1).astype(np.float32)
    X_val_flat = X_val.reshape(X_val.shape[0], -1).astype(np.float32)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float32)
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Apply PCA (subsample for fitting if very large to prevent memory OOM and speed up training)
    n_comp = min(n_components, X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_seed)
    
    if X_train_scaled.shape[0] > 4000:
        logger.info(f"Training set has {X_train_scaled.shape[0]} samples. Fitting PCA on a random subset of 4000 samples for efficiency...")
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(X_train_scaled.shape[0], size=4000, replace=False)
        pca.fit(X_train_scaled[indices])
        X_train_pca = pca.transform(X_train_scaled)
    else:
        logger.info(f"Fitting PCA with {n_comp} components on full train set...")
        X_train_pca = pca.fit_transform(X_train_scaled)
        
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    logger.info(f"PCA fitted. Total explained variance: {explained_variance:.2f}%")
    
    cnn_data = {
        'train': (X_train_cnn, y_train_onehot),
        'val': (X_val_cnn, y_val_onehot),
        'test': (X_test_cnn, y_test_onehot)
    }
    
    pca_data = {
        'train': (X_train_pca, y_train_onehot),
        'val': (X_val_pca, y_val_onehot),
        'test': (X_test_pca, y_test_onehot)
    }
    
    return cnn_data, pca_data, y_test, scaler, pca, class_names
