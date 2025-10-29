"""Data loading pipeline for BDD100K dataset."""
import os
from torch.utils.data import Dataset
from PIL import Image
import json


class BDD100KDataset(Dataset):
    """BDD100K dataset for object detection with text prompts.
    
    Supports both 10k subset and full 100k dataset with proper splits.
    """
    
    def __init__(self, data_dir, split="train", transform=None, max_samples=None, use_100k=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.use_100k = use_100k
        
        # Determine image directory based on dataset type
        if use_100k:
            # For 100k dataset, look for actual extracted structure: data/100k/100k/{split}/
            images_dir_candidates = [
                os.path.join(data_dir, "100k", "100k", split),  # Actual extracted structure
                os.path.join(data_dir, "images", "100k", split),
                os.path.join(data_dir, "100k", split),
                os.path.join(data_dir, split),
            ]
            self.images_dir = next((d for d in images_dir_candidates if os.path.exists(d)), None)
            if not self.images_dir:
                self.images_dir = os.path.join(data_dir, split)
                print(f"Warning: Assuming 100k images in {self.images_dir}")
        else:
            # For cleaned 10k dataset, look for: data/10k_clean/images/{split}/
            # Or fallback to: data/10k/{split}/ or data/{split}/
            images_dir_candidates = [
                os.path.join(data_dir, "images", split),  # Cleaned dataset structure
                os.path.join(data_dir, split),  # Direct structure
            ]
            self.images_dir = next((d for d in images_dir_candidates if os.path.exists(d)), None)
            if not self.images_dir:
                self.images_dir = os.path.join(data_dir, split)
                print(f"Warning: Assuming images in {self.images_dir}")
        
        # Try multiple annotation locations (consolidated JSON first, then per-image JSONs)
        if use_100k:
            # For 100k dataset, prefer consolidated JSON, fallback to per-image JSONs
            annotation_paths = [
                os.path.join(data_dir, "labels", f"bdd100k_labels_images_{split}.json"),
                os.path.join(data_dir, "annotations", f"bdd100k_labels_images_{split}.json"),
                os.path.join(data_dir, f"bdd100k_labels_images_{split}.json"),
            ]
        else:
            # For cleaned 10k dataset, look for consolidated JSON in labels directory
            annotation_paths = [
                os.path.join(data_dir, "labels", f"bdd100k_labels_{split}.json"),
                os.path.join(data_dir, "annotations", f"bdd100k_labels_{split}.json"),
                os.path.join(data_dir, f"bdd100k_labels_{split}.json"),
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
            # Fallback: per-image JSON directories
            if use_100k:
                # Original 100k dataset structure
                per_image_dir_candidates = [
                    os.path.join(data_dir, "labels", "100k", split),
                    os.path.join(os.path.dirname(data_dir), "labels", "100k", split),
                ]
            else:
                # Cleaned 10k dataset may have per-image JSONs in labels/{split}/ or labels/100k/{split}/
                per_image_dir_candidates = [
                    os.path.join(data_dir, "labels", split),  # Cleaned dataset structure
                    os.path.join(data_dir, "labels", "100k", split),  # Original structure
                    os.path.join(os.path.dirname(data_dir), "labels", "100k", split),
                ]
            per_image_dir = next((p for p in per_image_dir_candidates if os.path.isdir(p)), None)
            if per_image_dir:
                print(f"Found per-image labels in {per_image_dir}; matching to available images...")
                # Build set of available image basenames (without extension)
                try:
                    image_basenames = {f.replace('.jpg', '') for f in os.listdir(self.images_dir) if f.endswith('.jpg')}
                    image_files = {f.replace('.jpg', ''): f for f in os.listdir(self.images_dir) if f.endswith('.jpg')}
                except FileNotFoundError:
                    image_basenames = set()
                    image_files = {}
                
                # Build index: JSON name field -> JSON file path
                json_name_to_file = {}
                json_files = [f for f in os.listdir(per_image_dir) if f.endswith('.json')]
                
                print(f"Indexing {len(json_files)} JSON files by name field...")
                for fname in json_files:
                    json_path = os.path.join(per_image_dir, fname)
                    try:
                        with open(json_path, 'r') as jf:
                            ann = json.load(jf)
                        json_name = ann.get('name', '')
                        if json_name:
                            json_name_to_file[json_name] = json_path
                    except:
                        pass
                
                # Match images to JSONs using BOTH methods:
                # 1. Filename match: <image>.jpg -> <image>.json
                # 2. Name field match: image name matches JSON["name"] field
                matched = []
                matched_images = set()
                
                for img_base in image_basenames:
                    img_file = image_files.get(img_base, f"{img_base}.jpg")
                    
                    # Method 1: Try filename match
                    json_path = os.path.join(per_image_dir, f"{img_base}.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as jf:
                            ann = json.load(jf)
                    # Method 2: Try name field match
                    elif img_base in json_name_to_file:
                        json_path = json_name_to_file[img_base]
                        with open(json_path, 'r') as jf:
                            ann = json.load(jf)
                    else:
                        continue  # No match found
                    
                    # Extract objects from JSON structure
                    labels_list = []
                    if "frames" in ann and len(ann["frames"]) > 0:
                        for frame in ann["frames"]:
                            if "objects" in frame:
                                labels_list.extend(frame["objects"])
                    elif "labels" in ann:
                        labels_list = ann["labels"]
                    
                    matched.append({
                        "name": img_file,
                        "labels": labels_list,
                    })
                    matched_images.add(img_base)
                
                self.annotations = matched
                match_rate = 100 * len(matched_images) / len(image_basenames) if image_basenames else 0
                print(f"Matched {len(self.annotations)} per-image labels ({match_rate:.1f}% of {len(image_basenames)} images)")
                
                if match_rate < 50:
                    print(f"\n⚠️  Warning: Only {match_rate:.1f}% of images matched. Consider downloading consolidated JSON:")
                    if use_100k:
                        print(f"   bdd100k_labels_images_{self.split}.json from https://bdd-data.berkeley.edu/portal.html")
                    else:
                        print(f"   bdd100k_labels_{self.split}.json from https://bdd-data.berkeley.edu/portal.html")
                    print(f"   Place in: {os.path.join(data_dir, 'annotations')} or {os.path.join(os.path.dirname(data_dir), 'annotations')}")
            else:
                self.annotations = []
                print(f"Warning: No annotations file found. Tried: {annotation_paths}")
                print("You may need to download consolidated annotations: https://bdd-data.berkeley.edu/portal.html")
        
        if max_samples:
            self.annotations = self.annotations[:max_samples]
        
        # Note: _create_10k_splits() is only needed when combining raw data into 10k subset.
        # If using cleaned dataset (data/10k_clean), splits are already correct and shouldn't be re-split.
    
    def _create_10k_splits(self):
        """Create proper 10k splits: 7k train, 1.5k val, 1.5k test"""
        total_images = len(self.annotations)
        if total_images < 10000:
            print(f"Warning: Only {total_images} images available, cannot create full 10k splits")
            return
        
        # Take first 10k images and split them
        subset = self.annotations[:10000]
        
        if self.split == "train":
            self.annotations = subset[:7000]
        elif self.split == "val":
            self.annotations = subset[7000:8500]
        elif self.split == "test":
            self.annotations = subset[8500:10000]
        
        print(f"Created {self.split} split: {len(self.annotations)} images from 10k subset")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        # Ensure image filename has .jpg extension
        image_name = ann["name"]
        if not image_name.endswith('.jpg'):
            image_name = image_name + '.jpg'
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        # Extract bounding boxes and labels
        # Handle both consolidated JSON format (with frames) and per-image format
        boxes = []
        labels = []
        objects_list = []
        
        # Check if this is consolidated JSON format (has frames)
        if "frames" in ann and len(ann["frames"]) > 0:
            for frame in ann["frames"]:
                if "objects" in frame:
                    objects_list.extend(frame["objects"])
        elif "labels" in ann:
            objects_list = ann["labels"]
        
        # Extract boxes and categories from objects
        for obj in objects_list:
            if "box2d" in obj:
                box = obj["box2d"]
                boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                labels.append(obj["category"])
        
        # Load segmentation if available
        segmentation = None
        if self.use_100k:
            seg_path = image_path.replace(".jpg", ".png").replace("images", "labels").replace("100k", "seg")
            if os.path.exists(seg_path):
                try:
                    segmentation = Image.open(seg_path)
                except:
                    pass
        
        if self.transform is not None:
            image = self.transform(image)
            if segmentation is not None:
                segmentation = self.transform(segmentation)
        
        result = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": ann["name"]
        }
        
        if segmentation is not None:
            result["segmentation"] = segmentation
            
        return result


if __name__ == "__main__":
    # Example usage
    print("BDD100K Dataset class ready")
    print("Note: Download BDD100K data separately and organize in data/ directory")

