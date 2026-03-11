import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

# =============================================================================
# FishDataset Class: Handles YOLO-formatted fish image extraction and cropping
# =============================================================================
ImageNetDefaultTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

class FishDataset(Dataset):
    def __init__(self, root_dir = None, transform=None, sub_folder=None):
        """
        Initializes the dataset by setting up paths and identifying valid image files.
        root_dir: Path to the dataset folder containing 'images' and 'labels' subdirectories.
        transform: The transformation pipeline (e.g., ResNet normalization).
        """
        self.root_dir = root_dir
        if self.root_dir is not None:
            self.images_dir = os.path.join(root_dir, 'images', sub_folder) if sub_folder is not None else os.path.join(root_dir, 'images')
            self.filenames = [f.replace('.jpg', '') for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
            self.labels_dir = os.path.join(root_dir, 'labels', sub_folder) if sub_folder is not None else os.path.join(root_dir, 'labels')
        
        self.transform = transform if transform is not None else transforms.Compose([
                                                                                transforms.ToTensor(),
                                                                                transforms.Normalize(
                                                                                    mean=[0.485, 0.456, 0.406], 
                                                                                    std=[0.229, 0.224, 0.225]
                                                                                )
                                                                            ])

        
        # List all files and en
        # sure there is a correspondence between images and labels

    def load_data(self, root_path, sub_folder):
        self.root_dir = root_path
        self.images_dir = os.path.join(root_path, 'images', sub_folder) if sub_folder is not None else os.path.join(root_path, 'images')
        self.labels_dir = os.path.join(root_path, 'labels', sub_folder) if sub_folder is not None else os.path.join(root_path, 'labels')
        self.filenames = [f.replace('.jpg', '') for f in os.listdir(self.images_dir) if f.endswith('.jpg')]

        
    def __len__(self):
        # Returns the total number of images in the dataset
        return len(self.filenames)

    def _get_letterbox_crop(self, image, box_norm):
        """
        Extracts a bounding box from the image and applies a letterbox resize (maintaining aspect ratio).
        box_norm: [x_center, y_center, width, height] in normalized YOLO format.
        """
        img_w, img_h = image.size
        cx, cy, w, h = box_norm
        
        # Convert YOLO format (center x, center y, width, height) to absolute (left, top, right, bottom)
        left = (cx - w / 2) * img_w
        top = (cy - h / 2) * img_h
        right = (cx + w / 2) * img_w
        bottom = (cy + h / 2) * img_h

        # 1. Perform the initial crop based on coordinates
        crop = image.crop((left, top, right, bottom))
        
        # 2. Resize maintaining aspect ratio using a thumbnail approach
        crop.thumbnail((224, 224), Image.Resampling.LANCZOS)
        
        # 3. Add black padding to ensure the final output is exactly 224x224
        delta_w = 224 - crop.size[0]
        delta_h = 224 - crop.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(crop, padding, fill=(0, 0, 0))

    def __getitem__(self, idx):
        """
        Loads an image, parses its corresponding YOLO label file, and returns cropped fish tensors.
        """
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, f"{fname}.jpg")
        label_path = os.path.join(self.labels_dir, f"{fname}.txt")
        
        # Extract metadata from the filename (Expected format: LGH020002_frame_1981)
        parts = fname.split('_')
        video_name = parts[0]
        frame_id = parts[-1]

        image = Image.open(img_path).convert('RGB')
        
        crops = []
        metadata = []

        # Parse YOLO annotations (class_id x_center y_center width height)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = list(map(float, line.split()))
                    class_id = int(data[0])
                    bbox = data[1:] # [x_center, y_center, w, h]
                    
                    # Process individual fish crop with letterboxing
                    fish_crop = self._get_letterbox_crop(image, bbox)
                    
                    # Apply additional transformations (e.g., ToTensor, Normalize)
                    if self.transform:
                        fish_crop = self.transform(fish_crop)
                    
                    crops.append(fish_crop)
                    metadata.append({
                        "video": video_name,
                        "frame": frame_id,
                        "class_id": class_id,
                        "bbox_yolo": bbox
                    })

        # Return a stack of image tensors if fish were detected, otherwise return empty structures
        if crops:
            return torch.stack(crops), metadata
        else:
            return torch.empty(0), []