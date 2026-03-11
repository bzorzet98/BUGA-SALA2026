import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

class FishDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Ruta a 'Datasets' que contiene 'images' y 'labels'.
        transform: El pipeline de normalización de ResNet.
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.transform = transform
        
        # Listamos los archivos y aseguramos que tengan correspondencia
        self.filenames = [f.replace('.jpg', '') for f in os.listdir(self.images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.filenames)

    def _get_letterbox_crop(self, image, box_norm):
        """
        Realiza el crop y mantiene la proporción con padding (Paso 3 del pipeline).
        box_norm: [x_center, y_center, width, height] (Formato YOLO)
        """
        img_w, img_h = image.size
        cx, cy, w, h = box_norm
        
        # Convertir YOLO (centro, ancho, alto) a (left, top, right, bottom) relativo
        left = (cx - w / 2) * img_w
        top = (cy - h / 2) * img_h
        right = (cx + w / 2) * img_w
        bottom = (cy + h / 2) * img_h

        # 1. Recorte
        crop = image.crop((left, top, right, bottom))
        
        # 2. Resize manteniendo proporción
        crop.thumbnail((224, 224), Image.Resampling.LANCZOS)
        
        # 3. Padding para completar 224x224
        delta_w = 224 - crop.size[0]
        delta_h = 224 - crop.size[1]
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        return ImageOps.expand(crop, padding, fill=(0, 0, 0))

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, f"{fname}.jpg")
        label_path = os.path.join(self.labels_dir, f"{fname}.txt")
        
        # Extraer Metadata del nombre del archivo
        # Estructura: LGH020002_frame_1981
        parts = fname.split('_')
        video_name = parts[0]
        frame_id = parts[-1]

        image = Image.open(img_path).convert('RGB')
        
        crops = []
        metadata = []

        # Leer anotaciones de YOLO (clase x_center y_center width height)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = list(map(float, line.split()))
                    class_id = int(data[0])
                    bbox = data[1:] # [x_center, y_center, w, h]
                    
                    # Preprocesar imagen del pez individual
                    fish_crop = self._get_letterbox_crop(image, bbox)
                    
                    if self.transform:
                        fish_crop = self.transform(fish_crop)
                    
                    crops.append(fish_crop)
                    metadata.append({
                        "video": video_name,
                        "frame": frame_id,
                        "class_id": class_id,
                        "bbox_yolo": bbox
                    })

        # Retornamos un stack de tensores si hay peces, de lo contrario manejamos el vacío
        if crops:
            return torch.stack(crops), metadata
        else:
            return torch.empty(0), []