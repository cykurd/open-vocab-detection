# Zero-Shot Road Hazard Detection

CLIP-Transformer architecture for novel object localization in autonomous driving.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

1. **Download BDD100K 10k images**: Place under `data/10k/` with structure:
   ```
   data/10k/
     ├── train/  (images)
     ├── val/    (images)
     └── test/   (images)
   ```

2. **Download annotations JSON files** from [BDD100K Portal](https://bdd-data.berkeley.edu/portal.html):
   - Go to "Detection" → Download labels
   - Download `bdd100k_labels_images_train.json` and `bdd100k_labels_images_val.json`
   - Place them in either:
     - `data/10k/annotations/` (recommended), OR
     - `data/annotations/` (parent directory)

## Phase 1: Cross-Attention Mechanism Training

Training cross-attention layers and FF networks while keeping CLIP encoders frozen.

### Quick Test (Local)
```bash
python train_bdd100k.py --data_dir data/10k --split train --max_samples 16 --steps 4
```

