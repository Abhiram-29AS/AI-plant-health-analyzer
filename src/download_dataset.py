import os
import shutil
import logging
import kagglehub

logger = logging.getLogger("PlantDiseaseClassification")

def print_manual_instructions():
    """
    Prints a clear, high-visibility warning box containing step-by-step 
    instructions on how to manually set up the dataset.
    """
    instructions = """
========================================================================
[!] DATASET NOT FOUND OR KAGGLE AUTHENTICATION REQUIRED
========================================================================
We were unable to automatically download the PlantVillage dataset from Kaggle.
This can happen if:
- Kaggle credentials are not set up on your machine.
- There are network connection issues.
- The Kaggle API rate limit was hit.

Please follow these manual steps to set up the dataset:

1. Download the PlantVillage dataset manually from Kaggle:
   URL: https://www.kaggle.com/datasets/arjuntejaswi/plant-village
   
2. Extract the downloaded ZIP file.

3. Place the extracted folder containing the plant disease classes 
   directly inside the 'dataset/' directory in this project, so that 
   the directory structure looks exactly like this:
   
   project_plant dc/
   ├── dataset/
   │   ├── Pepper__bell___Bacterial_spot/
   │   ├── Pepper__bell___healthy/
   │   ├── Potato___Early_blight/
   │   ├── Potato___healthy/
   │   ├── Tomato___Bacterial_spot/
   │   └── ... (other plant disease classes)
   ├── config.yaml
   └── src/

4. Once the dataset is in place, re-run the training pipeline:
   python src/train.py
========================================================================
"""
    print(instructions)

def setup_dataset(dataset_path="dataset"):
    """
    Checks if the dataset exists locally. If not, attempts to download 
    automatically via kagglehub. Falls back to manual instructions on failure.
    """
    # Check if dataset path already exists and contains class directories
    if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
        # Check if the folder has subfolders (i.e. classes)
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if len(subdirs) > 0:
            logger.info(f"Dataset already exists in '{dataset_path}' containing {len(subdirs)} classes.")
            return True
            
    logger.info(f"Dataset not found in '{dataset_path}'. Attempting automatic download from Kaggle...")
    
    # Try to download the dataset using kagglehub
    try:
        # Create output dataset directory
        os.makedirs(dataset_path, exist_ok=True)
        
        # Download using kagglehub
        downloaded_path = kagglehub.dataset_download("arjuntejaswi/plant-village")
        logger.info(f"Successfully downloaded dataset files to cache: {downloaded_path}")
        
        # Determine where the actual class folders are (sometimes they are inside a subfolder like 'PlantVillage')
        source_dir = downloaded_path
        # Look if there is a 'PlantVillage' or 'plant-village' subfolder inside
        for folder_name in os.listdir(downloaded_path):
            full_sub_path = os.path.join(downloaded_path, folder_name)
            if os.path.isdir(full_sub_path) and folder_name.lower() in ["plantvillage", "plant_village"]:
                source_dir = full_sub_path
                break
        
        # Copy all class subdirectories from cache to project dataset/ folder
        copied_classes = 0
        for item in os.listdir(source_dir):
            src_item_path = os.path.join(source_dir, item)
            dst_item_path = os.path.join(dataset_path, item)
            
            if os.path.isdir(src_item_path):
                # Ensure we don't copy system or metadata files
                if item.startswith('.') or item.lower() in ["__macosx"]:
                    continue
                # If destination already exists, clear it
                if os.path.exists(dst_item_path):
                    shutil.rmtree(dst_item_path)
                shutil.copytree(src_item_path, dst_item_path)
                copied_classes += 1
                
        logger.info(f"Successfully copied {copied_classes} classes to local '{dataset_path}' folder.")
        return True
        
    except Exception as e:
        logger.error(f"Automatic download failed due to error: {e}")
        print_manual_instructions()
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_dataset()
