#!/usr/bin/env python3
"""
Sampling script for generative models
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from src.utils import tensor_to_pil_image, load_model, seed_everything
from custom_model import create_custom_model


def main(args):
    # Set seed for reproducibility
    seed_everything(args.seed)
    print(f"Seed set to: {args.seed}")
    
    save_dir = Path(args.save_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_tpu = False
    if str(args.device).lower() == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            use_tpu = True
            device = xm.xla_device()
            print(f"Using XLA device: {device}")
        except Exception as e:
            raise RuntimeError(f"TPU requested but torch_xla is unavailable: {e}")
    
    # Load model
    if not Path(args.ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if any(nfe <= 0 for nfe in args.nfe_list):
        raise ValueError("All NFE values must be positive integers")
    print(f"Loading model from: {args.ckpt_path}")
    model = load_model(args.ckpt_path, create_custom_model, str(device), args.config_path)
    model.eval()
    print(f"✓ Model loaded: {type(model).__name__}")
    
    num_batches = int(np.ceil(args.num_samples / args.batch_size))
    
    # Generate samples for each NFE
    for nfe in args.nfe_list:
        nfe_dir = save_dir / f"nfe={nfe}"
        nfe_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\nGenerating {args.num_samples} samples with NFE={nfe}...")
        
        for i in tqdm(range(num_batches), desc=f"NFE={nfe}"):
            sidx = i * args.batch_size
            eidx = min(sidx + args.batch_size, args.num_samples)
            B = eidx - sidx
            
            # Generate samples
            with torch.no_grad():
                samples = model.sample((B, 3, 64, 64), num_inference_timesteps=nfe, verbose=False)
            
            # Save images
            pil_images = tensor_to_pil_image(samples)
            for j, img in zip(range(sidx, eidx), pil_images):
                try:
                    img.save(nfe_dir / f"{j:04d}.png")
                except Exception:
                    pass
        
        print(f"✓ Saved to {nfe_dir}")
    
    print(f"\n✓ All samples saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from generative model")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/best_model.pt",
                       help="Path to model checkpoint (default: ./checkpoints/best_model.pt)")
    parser.add_argument("--config_path", type=str, help="if None, the default config path is model_config.json under the same directory as the checkpoint")
    parser.add_argument("--save_dir", type=str, default="./samples",
                       help="Directory to save samples (default: ./samples)")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for generation")
    parser.add_argument("--num_samples", type=int, default=1000, 
                       help="Total number of samples to generate")
    parser.add_argument("--nfe_list", type=int, nargs="+", default=[1, 2, 4], 
                       help="List of NFE (Number of Function Evaluations) values to test (evaluation at 1, 2, 4)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run sampling on (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    main(args)
