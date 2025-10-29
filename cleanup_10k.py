#!/usr/bin/env python3
"""
Clean up BDD100K dataset to create a clean 10k subset with matching labels and segmentation.

This script:
1. Finds the first 10k images from the 100k dataset
2. Ensures all have matching labels and segmentation
3. Creates clean train/val/test splits (7k/1.5k/1.5k)
4. Optionally removes unmatched images to save space
"""

import os
import json
import shutil
from pathlib import Path
import argparse


def find_matching_for_split(data_dir, split: str, max_images: int, require_segmentation=True):
    """Find up to N images for a specific split that have labels and optionally segmentation.

    Actual structure:
    - Images: data/100k/100k/{split}/*.jpg
    - Labels: data/labels/100k/{split}/*.json (per-image JSON files)
    - Segmentation: data/segmentation/bdd100k_seg_maps/labels/{split}/*.png (train/val only, optional)
    
    Args:
        require_segmentation: If False, only require labels (segmentation becomes optional)
    """
    # Try multiple possible image paths
    images_dir_candidates = [
        Path(data_dir) / "100k" / "100k" / split,
        Path(data_dir) / "images" / "100k" / split,
        Path(data_dir) / "100k" / split,
    ]
    images_dir = next((d for d in images_dir_candidates if d.exists()), None)
    
    if not images_dir:
        print(f"❌ Images directory not found for split '{split}'. Tried: {images_dir_candidates}")
        return []

    # Per-image JSON labels directory
    labels_dir = Path(data_dir) / "labels" / "100k" / split
    has_labels = labels_dir.exists()

    # Segmentation directory
    seg_dir_candidates = [
        Path(data_dir) / "segmentation" / "bdd100k_seg_maps" / "labels" / split,
        Path(data_dir) / "labels" / "seg" / split,
    ]
    seg_dir = next((d for d in seg_dir_candidates if d.exists()), None)
    has_seg = seg_dir is not None

    image_files = sorted([f for f in images_dir.glob("*.jpg")])
    print(f"🔍 [{split}] scanning {len(image_files)} images (labels={'yes' if has_labels else 'no'}, seg={'yes' if has_seg else 'no'})")

    matching = []
    for img_path in image_files:
        if len(matching) >= max_images:
            break
        img_name = img_path.name
        img_base = img_path.stem

        # Check if labels exist (per-image JSON)
        labels_json = labels_dir / f"{img_base}.json" if has_labels else None
        if has_labels and not (labels_json and labels_json.exists()):
            continue

        # Check if segmentation exists (try multiple naming patterns)
        seg_path = None
        if has_seg:
            # Segmentation files use _train_id suffix regardless of split folder
            # Also try finding by prefix match since naming might vary
            seg_candidates = [
                seg_dir / f"{img_base}_train_id.png",  # Most common pattern
                seg_dir / f"{img_base}_val_id.png",
                seg_dir / f"{img_base}_test_id.png",
                seg_dir / f"{img_base}.png",
            ]
            seg_path = next((p for p in seg_candidates if p.exists()), None)
            
            # If exact match fails, try prefix match (some files might have extra suffixes)
            if not seg_path:
                matching_segs = list(seg_dir.glob(f"{img_base}*.png"))
                if matching_segs:
                    seg_path = matching_segs[0]  # Take first match
            
            # Only require segmentation if explicitly requested
            if require_segmentation and not seg_path:
                continue

        # Load labels if available
        labels_data = None
        if labels_json:
            try:
                with open(labels_json, 'r') as f:
                    labels_data = json.load(f)
            except:
                continue

        matching.append({
            'image_path': img_path,
            'seg_path': seg_path,
            'labels': labels_data,
            'name': img_name
        })

    print(f"✅ [{split}] matched {len(matching)} images with available data")
    return matching


def create_clean_splits(per_split_matches, output_dir):
    """Create clean splits from per-split matches, preserving official split folders."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Targets: 7k train, 1.5k val, 1.5k test (take up to available)
    splits = {
        'train': per_split_matches.get('train', [])[:7000],
        'val': per_split_matches.get('val', [])[:1500],
        'test': per_split_matches.get('test', [])[:1500],
    }
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'seg' / split).mkdir(parents=True, exist_ok=True)
    
    # Copy files and create consolidated labels
    all_labels = []
    
    for split_name, images in splits.items():
        print(f"📁 Creating {split_name} split: {len(images)} images")
        
        split_labels = []
        for img_data in images:
            # Copy image
            dst_img = output_dir / 'images' / split_name / img_data['name']
            shutil.copy2(img_data['image_path'], dst_img)
            
            # Copy segmentation
            if img_data.get('seg_path'):
                dst_seg = output_dir / 'seg' / split_name / f"{img_data['image_path'].stem}.png"
                shutil.copy2(img_data['seg_path'], dst_seg)
            
            # Add to labels
            if img_data.get('labels') is not None:
                split_labels.append(img_data['labels'])
                all_labels.append(img_data['labels'])
        
        # Save split-specific labels
        if split_labels:
            with open(output_dir / 'labels' / f'bdd100k_labels_{split_name}.json', 'w') as f:
                json.dump(split_labels, f)
    
    # Save consolidated labels
    if all_labels:
        with open(output_dir / 'labels' / 'bdd100k_labels_all.json', 'w') as f:
            json.dump(all_labels, f)
    
    print(f"✅ Clean 10k dataset created in {output_dir}")
    print(f"   Train: {len(splits['train'])} images")
    print(f"   Val: {len(splits['val'])} images") 
    print(f"   Test: {len(splits['test'])} images")


def delete_unmatched_files(data_dir, per_split_matches):
    """Delete unmatched images and labels from original dataset, keeping only matched ones."""
    from pathlib import Path
    
    data_dir = Path(data_dir)
    
    # Build sets of files to keep
    keep_images = {}
    keep_labels = {}
    
    for split, matches in per_split_matches.items():
        keep_images[split] = {m['name'] for m in matches}
        keep_labels[split] = {m['labels']['name'] if m.get('labels') else None for m in matches}
        keep_labels[split].discard(None)  # Remove None values
    
    deletions = {'images': 0, 'labels': 0}
    
    for split in ['train', 'val', 'test']:
        # Delete unmatched images
        images_dir_candidates = [
            data_dir / "100k" / "100k" / split,
            data_dir / "images" / "100k" / split,
            data_dir / "100k" / split,
        ]
        images_dir = next((d for d in images_dir_candidates if d.exists()), None)
        
        if images_dir:
            keep_set = keep_images.get(split, set())
            for img_file in images_dir.glob("*.jpg"):
                if img_file.name not in keep_set:
                    img_file.unlink()
                    deletions['images'] += 1
        
        # Delete unmatched labels
        labels_dir = data_dir / "labels" / "100k" / split
        if labels_dir.exists():
            keep_set = keep_labels.get(split, set())
            for json_file in labels_dir.glob("*.json"):
                base = json_file.stem
                if base not in keep_set:
                    json_file.unlink()
                    deletions['labels'] += 1
    
    print(f"✅ Deleted {deletions['images']} unmatched images and {deletions['labels']} unmatched labels")
    print(f"   Kept {sum(len(v) for v in keep_images.values())} images and {sum(len(v) for v in keep_labels.values())} labels")


def main():
    parser = argparse.ArgumentParser(description="Clean up BDD100K dataset to create 10k subset")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to BDD100K root directory")
    parser.add_argument("--output_dir", type=str, default="data/10k_clean", help="Output directory for clean dataset")
    parser.add_argument("--max_images", type=int, default=10000, help="Maximum number of images to include")
    parser.add_argument("--remove_unmatched", action="store_true", help="Remove unmatched images from original dataset")
    
    args = parser.parse_args()
    
    print("🧹 BDD100K Cleanup Script")
    print(f"📂 Source: {args.data_dir}")
    print(f"📁 Output: {args.output_dir}")
    
    # Find matching images per official split
    # No segmentation required - only need labels and images
    # Target: 7k train, 1.5k val, 1.5k test = 10k total
    target_train = 7000
    target_val = 1500
    target_test = 1500
    
    per_split_matches = {
        'train': find_matching_for_split(args.data_dir, 'train', min(target_train, args.max_images), require_segmentation=False),
        'val': find_matching_for_split(args.data_dir, 'val', min(target_val, args.max_images), require_segmentation=False),
        'test': find_matching_for_split(args.data_dir, 'test', min(target_test, args.max_images), require_segmentation=False),
    }

    total_found = sum(len(v) for v in per_split_matches.values())
    print(f"\n📊 Summary:")
    print(f"   Train: {len(per_split_matches['train'])} images")
    print(f"   Val: {len(per_split_matches['val'])} images")
    print(f"   Test: {len(per_split_matches['test'])} images")
    print(f"   Total: {total_found} images")
    
    if total_found < 10000:
        print(f"⚠️  Warning: Only found {total_found} images with available data across splits")
        # Continue anyway to create what we can
    
    # Create clean splits preserving split membership
    create_clean_splits(per_split_matches, args.output_dir)
    
    if args.remove_unmatched:
        print("\n🗑️  Starting deletion of unmatched files...")
        delete_unmatched_files(args.data_dir, per_split_matches)


if __name__ == "__main__":
    main()
