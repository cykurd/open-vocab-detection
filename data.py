"""Data loading pipeline for BDD100K dataset."""
import os
from torch.utils.data import Dataset
from PIL import Image
import json


class BDD100KDataset(Dataset):
    """BDD100K dataset for object detection with text prompts."""
    
    def __init__(self, data_dir, split="train", transform=None, max_samples=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.images_dir = os.path.join(data_dir, "images", split)
        self.annotations_file = os.path.join(data_dir, "annotations", f"bdd100k_labels_{split}.json")
        
        # Load annotations
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = []
            print(f"Warning: Annotations file not found at {self.annotations_file}")
        
        if max_samples:
            self.annotations = self.annotations[:max_samples]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.images_dir, ann["name"])
        image = Image.open(image_path).convert("RGB")
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        for obj in ann.get("labels", []):
            if "box2d" in obj:
                box = obj["box2d"]
                boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                labels.append(obj["category"])
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": ann["name"]
        }


if __name__ == "__main__":
    # Example usage
    print("BDD100K Dataset class ready")
    print("Note: Download BDD100K data separately and organize in data/ directory")

