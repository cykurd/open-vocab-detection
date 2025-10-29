import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from load_detector import load_detector
from data import BDD100KDataset
from losses import compute_detection_loss


def freeze_for_phase1(model: torch.nn.Module) -> int:
	trainable_params = 0
	for name, param in model.named_parameters():
		# Phase 1: Only allow decoder cross-attention and small FF layers to train
		if name.startswith("model.decoder") and (
			"encoder_attn" in name or "fc" in name or "final_layer_norm" in name
		):
			param.requires_grad = True
			trainable_params += param.numel()
		else:
			param.requires_grad = False
	return trainable_params


def collate_fn(batch):
	# Keep it simple: batch size 1 expected in this smoke test
	return batch[0]


def main():
	parser = argparse.ArgumentParser(description="Minimal BDD100K training smoke test")
	parser.add_argument("--data_dir", type=str, default="data", help="Path to BDD100K root (e.g., data with images/100k/ structure)")
	parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
	parser.add_argument("--max_samples", type=int, default=16, help="Limit number of samples for quick test")
	parser.add_argument("--batch_size", type=int, default=1, help="Use 1 for simplest run")
	parser.add_argument("--steps", type=int, default=4, help="Number of optimization steps")
	parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
	parser.add_argument("--use_10k", action="store_true", help="Use 10k subset instead of full 100k dataset")
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Force device selection (default: auto)")
	args = parser.parse_args()

	if args.device == "cpu":
		device = torch.device("cpu")
	elif args.device == "cuda":
		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	elif args.device == "mps":
		device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
	print(f"Using device: {device}")

	# Dataset
	try:
		dataset = BDD100KDataset(data_dir=args.data_dir, split=args.split, transform=None, max_samples=args.max_samples, use_100k=not args.use_10k)
		if len(dataset) == 0:
			print("No samples found. Common issues:")
			if not args.use_10k:
				print("1. Images: Ensure data_dir/images/100k/<split>/ contains images")
				print("2. Annotations: Download consolidated JSON from http://bdd-data.berkeley.edu/download.html")
				print("   Place as: data_dir/labels/bdd100k_labels_images_<split>.json")
			else:
				print("1. Images: Ensure data_dir/<split>/ contains images (e.g., data/10k/train/)")
				print("2. Annotations: Download JSON labels from http://bdd-data.berkeley.edu/download.html")
				print("   Place them as: data_dir/annotations/bdd100k_labels_<split>.json")
				print(f"   Or in parent directory: data/annotations/bdd100k_labels_{args.split}.json")
			return
		
		# Test loading a single sample to catch data issues early
		print("Testing data loading...")
		try:
			test_sample = dataset[0]
			print(f"✓ Sample loaded: image shape {test_sample['image'].size}, {len(test_sample.get('boxes', []))} boxes, {len(test_sample.get('labels', []))} labels")
		except Exception as e:
			print(f"Error loading test sample: {e}")
			import traceback
			traceback.print_exc()
			return
		
		dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
		
	except Exception as e:
		print(f"Error loading dataset: {e}")
		print("Please check your data directory and file structure.")
		return

	# Model
	try:
		print("Loading detector model...")
		detector, processor = load_detector()
		detector = detector.to(device)
		print("✓ Model loaded successfully")

		trainable = freeze_for_phase1(detector)
		total = sum(p.numel() for p in detector.parameters())
		print(f"Trainable params (Phase 1): {trainable:,} / {total:,}")

		optimizer = torch.optim.Adam([p for p in detector.parameters() if p.requires_grad], lr=args.lr)
		detector.train()
		print("✓ Model ready for training")
		
	except Exception as e:
		print(f"Error loading model: {e}")
		return

	# Create output directory
	output_dir = "outputs"
	checkpoint_dir = os.path.join(output_dir, "checkpoints")
	os.makedirs(checkpoint_dir, exist_ok=True)
	
	step = 0
	best_loss = float('inf')
	
	for sample in dataloader:
		if step >= args.steps:
			break
		
		try:
			image = sample["image"]
            # can update 'object' here to be more specific to the road hazards or any other prompt we want to use
			text = " ".join(sorted(set(sample.get("labels", ["object"])))).strip() or "object"
			
			# Process inputs with error handling
			inputs = processor(images=image, text=text, return_tensors="pt")
			inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

			# Forward pass with error handling
			outputs = detector(**inputs)
			loss = compute_detection_loss(outputs, sample.get("boxes", []), sample.get("labels", []))

			# Backward pass with gradient clipping
			optimizer.zero_grad()
			loss.backward()
			
			# Gradient clipping to prevent exploding gradients
			torch.nn.utils.clip_grad_norm_([p for p in detector.parameters() if p.requires_grad], max_norm=1.0)
			
			optimizer.step()

			step += 1
			current_loss = loss.item()
			print(f"Step {step}/{args.steps} - loss: {current_loss:.4f}")
			
			# Save checkpoint every 10 steps or if it's the best loss so far
			if step % 10 == 0 or current_loss < best_loss:
				checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
				torch.save({
					'step': step,
					'model_state_dict': detector.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': current_loss,
					'args': args
				}, checkpoint_path)
				print(f"  Saved checkpoint: {checkpoint_path}")
				
				if current_loss < best_loss:
					best_loss = current_loss
					best_path = os.path.join(checkpoint_dir, "best_model.pth")
					torch.save({
						'step': step,
						'model_state_dict': detector.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'loss': current_loss,
						'args': args
					}, best_path)
					print(f"  New best model saved: {best_path}")
			
		except Exception as e:
			print(f"Error on step {step + 1}: {e}")
			print(f"Sample keys: {list(sample.keys())}")
			print(f"Image shape: {sample['image'].size if 'image' in sample else 'N/A'}")
			print(f"Labels: {sample.get('labels', 'N/A')}")
			print(f"Boxes: {sample.get('boxes', 'N/A')}")
			break

	# Save final model
	final_path = os.path.join(checkpoint_dir, "final_model.pth")
	torch.save({
		'step': step,
		'model_state_dict': detector.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': current_loss if 'current_loss' in locals() else float('inf'),
		'args': args
	}, final_path)
	print(f"Final model saved: {final_path}")
	print("Training completed.")


if __name__ == "__main__":
	main()
