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
   - Structure will be:
     ```
     data/
       ├── 100k/100k/
       │   ├── train/  (70k images)
       │   ├── val/    (10k images)
       │   └── test/   (20k images)
       ├── labels/100k/
       │   ├── train/  (*.json files)
       │   ├── val/    (*.json files)
       │   └── test/   (*.json files)
       └── segmentation/  (optional)
     ```

3. **Clean up dataset** (optional, creates 10k subset):
   ```bash
   python cleanup_10k.py --data_dir data --output_dir data/10k_clean --max_images 10000 --remove_unmatched
   ```
   This creates `data/10k_clean/` with:
   - 7,000 train images + labels
   - 1,500 val images + labels  
   - 1,500 test images + labels
   And removes unmatched files from original dataset to save space.

## Phase 1: Cross-Attention Mechanism Training

Training cross-attention layers and FF networks while keeping CLIP encoders frozen.

### Quick Test (Local)
```bash
python train_bdd100k.py --data_dir data --split train --max_samples 16 --steps 4
```

