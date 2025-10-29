# Zero-Shot Road Hazard Detection

CLIP-Transformer architecture for novel object localization in autonomous driving.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

1. **Download from [BDD100K](http://bdd-data.berkeley.edu/download.html)**:
   - Click "100K Images" button
   - Click "Labels" button  
   - Click "Segmentations" button

2. **Extract to `data/` folder**:
   - Unzip all downloaded files
   - Rename folders if needed to match structure below
   - Final structure should be:
     ```
     data/
       ├── images/100k/
       │   ├── train/  (70k images)
       │   ├── val/    (10k images)
       │   └── test/   (20k images)
       ├── labels/
       │   ├── bdd100k_labels_images_train.json
       │   └── bdd100k_labels_images_val.json
       └── labels/seg/  (segmentation masks)
     ```

3. **Clean up dataset** (optional):
   ```bash
   python cleanup_10k.py  # Creates clean 10k subset with matching labels
   ```

## Phase 1: Cross-Attention Mechanism Training

Training cross-attention layers and FF networks while keeping CLIP encoders frozen.

### Quick Test (Local)
```bash
python train_bdd100k.py --data_dir data --split train --max_samples 16 --steps 4
```

