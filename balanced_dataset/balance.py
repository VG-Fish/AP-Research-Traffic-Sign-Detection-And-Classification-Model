from json import load
from typing import Dict, Any, Set, List, Tuple
from os import makedirs
from os.path import abspath, exists
from random import choice, sample
from shutil import copy, rmtree
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
BALANCED_DATASET_DIRECTORY = "balanced_augmented_mapillary_dataset"
MAPILLARY_DATASET_DIRECTORY = "mapillary_dataset"
CLASS_INFO_FILE = "class_information.json"

# Dataset split configuration
SPLITS = {
    "minority": {
        0.5: 400,
        0.75: 920,
        1.0: 2140,
    },
    "majority": {
        2: 130,
        4: 200,
        12: 50,
        16: 50,
        20: 30,
    },
    "background": 80,
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}


def create_directories(exist_ok: bool = False) -> None:
    """Create the necessary directory structure for the balanced dataset."""
    logger.info("Creating directory structure...")
    
    directories = ["train", "val", "test"]
    for directory in directories:
        dir_path = f"{BALANCED_DATASET_DIRECTORY}/{directory}"
        if exists(dir_path):
            logger.info(f"Removing existing directory: {dir_path}")
            rmtree(dir_path)
        
        makedirs(f"{dir_path}/images", exist_ok=exist_ok)
        makedirs(f"{dir_path}/labels", exist_ok=exist_ok)
    
    logger.info("Directory structure created successfully.")


def load_data() -> Dict[str, Any]:
    """Load class information from JSON file."""
    logger.info(f"Loading data from {CLASS_INFO_FILE}...")
    try:
        with open(CLASS_INFO_FILE, "r") as f:
            data = load(f)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logger.error(f"Error: {CLASS_INFO_FILE} not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def find_image_location(image_id: str) -> Tuple[str, str]:
    """Find the location of an image in the original dataset.
    
    Args:
        image_id: The ID of the image to find
        
    Returns:
        Tuple containing paths to the image and its label, or (None, None) if not found
    """
    splits = ["train", "val", "test"]
    
    for split in splits:
        img_path = f"{MAPILLARY_DATASET_DIRECTORY}/{split}/images/{image_id}.jpg"
        label_path = f"{MAPILLARY_DATASET_DIRECTORY}/{split}/labels/{image_id}.txt"
        
        if exists(img_path):
            return img_path, label_path
    
    return None, None


def copy_image_and_label(image_id: str, destination_dir: str) -> bool:
    """Copy an image and its label to the destination directory.
    
    Args:
        image_id: The ID of the image to copy
        destination_dir: The destination directory (train/val/test)
        
    Returns:
        Boolean indicating if the copy was successful
    """
    source_img_path, source_label_path = find_image_location(image_id)
    
    if not source_img_path:
        return False
    
    destination_img_path = f"{BALANCED_DATASET_DIRECTORY}/{destination_dir}/images/{image_id}.jpg"
    destination_label_path = f"{BALANCED_DATASET_DIRECTORY}/{destination_dir}/labels/{image_id}.txt"
    
    try:
        copy(abspath(source_img_path), abspath(destination_img_path))
        
        # Only copy label if it exists and we're not in test split
        if source_label_path and exists(source_label_path) and destination_dir != "test":
            copy(abspath(source_label_path), abspath(destination_label_path))
        
        return True
    except Exception as e:
        logger.error(f"Error copying {image_id}: {e}")
        return False


def sample_images(image_list: List[str], count: int, used_images: Set[str], high_diversity: bool = False) -> List[str]:
    """Sample images from a list, optionally ensuring diversity.
    
    Args:
        image_list: List of image IDs to sample from
        count: Number of images to sample
        used_images: Set of already used image IDs
        high_diversity: If True, ensure all sampled images are unique from used_images
        
    Returns:
        List of sampled image IDs
    """
    available_images = len(image_list)
    if count > available_images:
        logger.warning(f"Requested {count} images but only {available_images} are available")
        count = available_images
    
    if high_diversity:
        # For high diversity, remove already used images from consideration
        available_list = [img for img in image_list if img not in used_images]
        
        # If not enough unique images, warn and use what's available
        if len(available_list) < count:
            logger.warning(f"Not enough unique images available. Requested {count}, available {len(available_list)}")
            count = len(available_list)
        
        return sample(available_list, count) if available_list else []
    else:
        # For regular sampling, just randomly select from the full list
        return [choice(image_list) for _ in range(count)]


def distribute_images(data: Dict[str, Any], class_type: str, dataset_splits: Dict[str, float]) -> Set[str]:
    """Distribute images of a specific class type according to the dataset splits.
    
    Args:
        data: The loaded class information data
        class_type: The type of class ('minority', 'majority', or 'background')
        dataset_splits: Dictionary mapping split names to their proportions
        
    Returns:
        Set of used image IDs
    """
    logger.info(f"Distributing {class_type} class images...")
    used_images: Set[str] = set()
    
    if class_type == "background":
        # Handle background images (fixed number)
        total_bg_images = SPLITS["background"]
        background_data = data.get("background_class_images", [])
        
        for directory, percent in dataset_splits.items():
            num_images = int(total_bg_images * percent)
            selected_images = sample_images(background_data, num_images, used_images)
            
            success_count = 0
            for img_id in selected_images:
                if copy_image_and_label(img_id, directory):
                    used_images.add(img_id)
                    success_count += 1
            
            logger.info(f"Added {success_count} background images to {directory}")
        
        return used_images
    
    # Handle minority or majority class images
    for bound, amount in SPLITS[class_type].items():
        class_data = data[f"{class_type}_class_bounds"][str(bound)]
        logger.info(f"Processing {class_type} class with bound {bound}, {len(class_data)} images available")
        
        for directory, percent in dataset_splits.items():
            num_images = int(amount * percent)
            
            # Use high diversity sampling for higher bounds
            high_diversity = bound / amount > 0.6 if isinstance(bound, (int, float)) else False
            selected_images = sample_images(class_data, num_images, used_images, high_diversity)
            
            success_count = 0
            for img_id in selected_images:
                if copy_image_and_label(img_id, directory):
                    used_images.add(img_id)
                    success_count += 1
            
            logger.info(f"Added {success_count}/{num_images} {class_type} images (bound {bound}) to {directory}")
    
    return used_images


def create_dataset() -> None:
    """Create the balanced dataset according to the specified configuration."""
    dataset_splits = {
        "train": SPLITS["train"],
        "val": SPLITS["val"], 
        "test": SPLITS["test"],
    }

    # Validate configuration
    total = sum(dataset_splits.values())
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Dataset splits must sum to 1.0, got {total}")
    
    for required in ["majority", "minority", "background"]:
        if required not in SPLITS:
            raise ValueError(f"Missing required split configuration for '{required}'")

    # Load data and distribute images
    data = load_data()
    
    # Process each class type
    distribute_images(data, "minority", dataset_splits)
    distribute_images(data, "majority", dataset_splits)
    distribute_images(data, "background", dataset_splits)
    
    # Remove test labels as specified in original code
    test_labels_dir = f"{BALANCED_DATASET_DIRECTORY}/test/labels"
    if exists(test_labels_dir):
        logger.info(f"Removing test labels directory: {test_labels_dir}")
        rmtree(test_labels_dir)
    
    logger.info("Dataset creation completed successfully.")


def main() -> None:
    """Main function to run the dataset creation process."""
    try:
        logger.info("Starting balanced dataset creation process")
        create_directories(exist_ok=True)
        create_dataset()
        logger.info("Balanced dataset created successfully!")
    except Exception as e:
        logger.error(f"Failed to create balanced dataset: {e}")
        raise


if __name__ == "__main__":
    main()