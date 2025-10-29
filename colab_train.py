import os
import sys
import argparse
import subprocess


def in_colab() -> bool:
	try:
		import google.colab  # type: ignore
		return True
	except Exception:
		return False


def try_mount_drive():
	if not in_colab():
		return False
	try:
		from google.colab import drive  # type: ignore
		drive.mount('/content/drive', force_remount=False)
		print("✓ Google Drive mounted at /content/drive")
		return True
	except Exception as e:
		print(f"Warning: Could not mount Google Drive: {e}")
		return False


def pip_install_requirements(project_root: str):
	req_path = os.path.join(project_root, 'requirements.txt')
	if not os.path.exists(req_path):
		print(f"No requirements.txt found at {req_path}, skipping installs")
		return
	print("Installing Python dependencies (requirements.txt)...")
	subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])


def main():
	parser = argparse.ArgumentParser(description="Colab launcher for BDD100K training")
	parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/data/10k_clean",
		help="Path to dataset root on Drive or local (default: /content/drive/MyDrive/data/10k_clean)")
	parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
	parser.add_argument("--max_samples", type=int, default=64, help="Limit number of samples for quick GPU smoke test")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
	parser.add_argument("--steps", type=int, default=20, help="Number of optimization steps")
	parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
	parser.add_argument("--use_10k", action="store_true", help="Use 10k subset layout (e.g., data/10k_clean)")
	parser.add_argument("--skip_mount", action="store_true", help="Do not attempt to mount Google Drive")
	args = parser.parse_args()

	project_root = os.path.dirname(os.path.abspath(__file__))
	print(f"Project root: {project_root}")

	if not args.skip_mount:
		try_mount_drive()

	# Ensure dependencies
	pip_install_requirements(project_root)

	# Prefer CUDA on Colab if available
	device = "cuda"

	# Build command to run the existing training script
	train_script = os.path.join(project_root, "train_bdd100k.py")
	cmd = [
		sys.executable,
		train_script,
		"--data_dir", args.data_dir,
		"--split", args.split,
		"--max_samples", str(args.max_samples),
		"--batch_size", str(args.batch_size),
		"--steps", str(args.steps),
		"--lr", str(args.lr),
		"--device", device,
	]
	if args.use_10k:
		cmd.append("--use_10k")

	print("Launching training:")
	print(" ", " ".join(cmd))
	res = subprocess.run(cmd, check=False)
	if res.returncode != 0:
		print(f"Training script exited with code {res.returncode}")
		sys.exit(res.returncode)

	print("Colab training run finished.")


if __name__ == "__main__":
	main()


