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
        
        # Try multiple possible structures
        # Structure 1: data/10k/{split}/ (images directly in split folder)
        if os.path.exists(os.path.join(data_dir, split)):
            self.images_dir = os.path.join(data_dir, split)
        # Structure 2: data/10k/images/{split}/ (standard structure)
        elif os.path.exists(os.path.join(data_dir, "images", split)):
            self.images_dir = os.path.join(data_dir, "images", split)
        else:
            self.images_dir = os.path.join(data_dir, split)
            print(f"Warning: Assuming images in {self.images_dir}")
        
        # Try multiple annotation locations
        annotation_paths = [
            os.path.join(data_dir, "annotations", f"bdd100k_labels_{split}.json"),
            os.path.join(data_dir, f"labels_{split}.json"),
            os.path.join(data_dir, f"bdd100k_labels_{split}.json"),
            os.path.join(os.path.dirname(data_dir), "annotations", f"bdd100k_labels_{split}.json"),
        ]
        
        self.annotations_file = None
        for path in annotation_paths:
            if os.path.exists(path):
                self.annotations_file = path
                break
        
        # Load annotations
        if self.annotations_file and os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
            print(f"Loaded {len(self.annotations)} annotations from {self.annotations_file}")
        else:
            self.annotations = []
            print(f"Warning: No annotations file found. Tried: {annotation_paths}")
            print("You may need to download annotations separately from: https://bdd-data.berkeley.edu/portal.html")
        
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

