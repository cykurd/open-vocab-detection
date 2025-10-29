#!/usr/bin/env python3
"""
Single Colab script for Zero-Shot Road Hazard Detection training.
Downloads repo, installs dependencies, and runs training on BDD100K 10k subset.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True, shell=False):
    """Run a command and print output."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


def setup_environment():
    """Install required packages."""
    print("🔧 Installing dependencies...")
    
    # Install PyTorch with CUDA support
    run_command([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    
    # Install other requirements
    run_command([
        sys.executable, "-m", "pip", "install",
        "transformers", "matplotlib", "pillow", "numpy", "opencv-python"
    ])


def download_repo():
    """Clone the repository."""
    print("📥 Downloading repository...")
    
    repo_url = "https://github.com/cykurd/open-vocab-detection.git"
    repo_dir = "/content/open-vocab-detection"
    
    if os.path.exists(repo_dir):
        print("Repository already exists, updating...")
        run_command(["git", "pull"], cwd=repo_dir)
    else:
        run_command(["git", "clone", repo_url, repo_dir])
    
    return repo_dir


def setup_data_structure(data_dir):
    """Create expected data structure and provide instructions."""
    print("📁 Setting up data structure...")
    
    # Create expected directories
    os.makedirs(f"{data_dir}/10k_clean/images/train", exist_ok=True)
    os.makedirs(f"{data_dir}/10k_clean/images/val", exist_ok=True)
    os.makedirs(f"{data_dir}/10k_clean/images/test", exist_ok=True)
    os.makedirs(f"{data_dir}/10k_clean/labels", exist_ok=True)
    
    print(f"""
📋 DATA UPLOAD INSTRUCTIONS:
1. Upload your 10k_clean dataset to: {data_dir}/10k_clean/
   
   Expected structure:
   {data_dir}/10k_clean/
   ├── images/
   │   ├── train/     (7,000 images)
   │   ├── val/       (1,500 images) 
   │   └── test/      (1,500 images)
   └── labels/
       ├── bdd100k_labels_train.json
       ├── bdd100k_labels_val.json
       └── bdd100k_labels_test.json

2. Or if you have the raw 100k dataset, place it in: {data_dir}/
   and the script will run cleanup_10k.py to create the 10k subset.

Press Enter when data is uploaded, or Ctrl+C to exit.
""")
    
    input("Press Enter to continue...")


def run_training(repo_dir, data_dir, args):
    """Run the training script."""
    print("🚀 Starting training...")
    
    # Change to repo directory
    os.chdir(repo_dir)
    
    # Build training command
    cmd = [
        sys.executable, "train_bdd100k.py",
        "--data_dir", data_dir,
        "--split", args.split,
        "--max_samples", str(args.max_samples),
        "--batch_size", str(args.batch_size),
        "--steps", str(args.steps),
        "--lr", str(args.lr),
        "--device", "cuda",
        "--use_10k"
    ]
    
    print(f"Training command: {' '.join(cmd)}")
    
    # Run training
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"❌ Training failed with exit code {result.returncode}")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Single Colab script for BDD100K training")
    parser.add_argument("--data_dir", type=str, default="/content/data", 
                       help="Path to data directory (default: /content/data)")
    parser.add_argument("--split", type=str, default="train", 
                       choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=1000, 
                       help="Maximum number of samples to use")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Batch size")
    parser.add_argument("--steps", type=int, default=100, 
                       help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-5, 
                       help="Learning rate")
    parser.add_argument("--skip_setup", action="store_true", 
                       help="Skip environment setup (if already done)")
    
    args = parser.parse_args()
    
    print("🎯 Zero-Shot Road Hazard Detection - Colab Training")
    print("=" * 60)
    
    # Setup environment
    if not args.skip_setup:
        setup_environment()
    
    # Download repository
    repo_dir = download_repo()
    
    # Setup data structure and wait for upload
    setup_data_structure(args.data_dir)
    
    # Check if data exists
    if not os.path.exists(f"{args.data_dir}/10k_clean"):
        print("❌ No 10k_clean dataset found!")
        print("Please upload your data and run again.")
        return 1
    
    # Run training
    exit_code = run_training(repo_dir, args.data_dir, args)
    
    print("\n" + "=" * 60)
    print("🏁 Training session complete!")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
