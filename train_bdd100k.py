import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from load_detector import load_detector
from data import BDD100KDataset


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


def compute_naive_loss(outputs, target_boxes):
	# NOTE: This is a tiny smoke-test loss, not the final training objective
	pred_boxes = outputs.pred_boxes  # [B, 900, 4]
	objectness = outputs.logits      # [B, 900, 256]
	
	# Stabilize objectness loss - clamp to prevent -inf
	objectness_clamped = torch.clamp(objectness, min=-10.0, max=10.0)
	loss_obj = objectness_clamped.mean()
	
	# Handle bounding box loss more carefully
	if len(target_boxes) > 0:
		tb = torch.tensor(target_boxes, dtype=torch.float32, device=pred_boxes.device)
		K = min(tb.shape[0], pred_boxes.shape[1])
		# Clamp predictions to reasonable range
		pred_boxes_clamped = torch.clamp(pred_boxes[0, :K], min=-10.0, max=10.0)
		loss_box = nn.functional.l1_loss(pred_boxes_clamped, tb[:K])
	else:
		loss_box = torch.zeros((), device=pred_boxes.device)
	
	total_loss = loss_obj + loss_box
	
	# Final safety check
	if torch.isnan(total_loss) or torch.isinf(total_loss):
		print(f"Warning: Invalid loss detected (obj: {loss_obj.item():.4f}, box: {loss_box.item():.4f})")
		total_loss = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
	
	return total_loss


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

	step = 0
	for sample in dataloader:
		if step >= args.steps:
			break
		
		try:
			image = sample["image"]
			text = " ".join(sorted(set(sample.get("labels", ["object"])))).strip() or "object"
			
			# Process inputs with error handling
			inputs = processor(images=image, text=text, return_tensors="pt")
			inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

			# Forward pass with error handling
			outputs = detector(**inputs)
			loss = compute_naive_loss(outputs, sample.get("boxes", []))

			# Backward pass with gradient clipping
			optimizer.zero_grad()
			loss.backward()
			
			# Gradient clipping to prevent exploding gradients
			torch.nn.utils.clip_grad_norm_([p for p in detector.parameters() if p.requires_grad], max_norm=1.0)
			
			optimizer.step()

			step += 1
			print(f"Step {step}/{args.steps} - loss: {loss.item():.4f}")
			
		except Exception as e:
			print(f"Error on step {step + 1}: {e}")
			print(f"Sample keys: {list(sample.keys())}")
			print(f"Image shape: {sample['image'].size if 'image' in sample else 'N/A'}")
			print(f"Labels: {sample.get('labels', 'N/A')}")
			print(f"Boxes: {sample.get('boxes', 'N/A')}")
			break

	print("Smoke test completed.")


if __name__ == "__main__":
	main()
