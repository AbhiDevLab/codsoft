import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    """
    Load all images from a folder and its subfolders.
    
    Args:
        folder: Path to the folder
        
    Returns:
        Tuple of (images, labels)
    """
    images = []
    labels = []
    
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        
        if not os.path.isdir(person_folder):
            continue
            
        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            
            if not os.path.isfile(img_path):
                continue
                
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(person_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images, labels

def display_image(image, title=None, figsize=(10, 8)):
    """
    Display an image using matplotlib.
    
    Args:
        image: Image to display
        title: Optional title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Convert BGR to RGB for display
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
