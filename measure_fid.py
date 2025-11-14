#!/usr/bin/env python3
"""
FID measurement script using clean-fid
Compute FID scores between generated samples and reference dataset
"""

import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm

from cleanfid import fid
from dataset import SimpsonsDataModule
from src.utils import tensor_to_pil_image


def prepare_reference_images(data_root, reference_dir, num_samples=None):
    """
    Prepare reference images from validation dataset.
    
    Args:
        data_root: Root directory of the dataset
        reference_dir: Directory to save reference images
        num_samples: Number of samples to save (None = all validation images)
        
    Returns:
        Path to reference directory
    """
    print(f"Preparing reference images from validation set...")
    
    # Load dataset
    data_module = SimpsonsDataModule(root=data_root, batch_size=32, num_workers=4)
    val_loader = data_module.val_dataloader()
    
    # Create reference directory
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    
    # Save validation images
    count = 0
    for batch in tqdm(val_loader, desc="Saving reference images"):
        if num_samples and count >= num_samples:
            break
        
        # Convert to PIL images
        pil_images = tensor_to_pil_image(batch)
        
        for img in pil_images:
            if num_samples and count >= num_samples:
                break
            img.save(reference_dir / f"{count:04d}.png")
            count += 1
    
    print(f"✓ Saved {count} reference images to {reference_dir}")
    return str(reference_dir)


def compute_fid(generated_dir, reference_dir, device="cuda", batch_size=64):
    """
    Compute FID score using clean-fid.
    
    Args:
        generated_dir: Directory containing generated images
        reference_dir: Directory containing reference images
        device: Device to use for computation
        batch_size: Batch size for FID computation
    
    Returns:
        FID score
    """
    print(f"\nComputing FID...")
    print(f"  Generated: {generated_dir}")
    print(f"  Reference: {reference_dir}")
    
    fid_score = fid.compute_fid(
        generated_dir, 
        reference_dir, 
        device=device,
        batch_size=batch_size,
        num_workers=4
    )
    
    return fid_score


def main():
    parser = argparse.ArgumentParser(description="Measure FID scores using clean-fid")
    
    # Input directories
    parser.add_argument("--generated_dir", type=str, required=True,
                       help="Directory containing generated image files")
    parser.add_argument("--reference_dir", type=str, default="./data/simpsons_64x64/val",
                       help="Directory containing reference images (will be created from validation set if not exists)")
    
    # Reference preparation (only used if reference_dir doesn't exist)
    parser.add_argument("--data_root", type=str, default="./data/datasets/kostastokis/simpsons-faces/versions/1/cropped",
                       help="Root directory of Simpsons dataset (for creating reference images if needed)")
    
    # Computation settings
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for FID computation")
    
    # Output
    parser.add_argument("--output_path", type=str, default="./fid_results.json",
                       help="Path to save FID results as JSON")
    
    args = parser.parse_args()
    
    # Check generated directory exists
    if not os.path.exists(args.generated_dir):
        raise ValueError(f"Generated directory not found: {args.generated_dir}")
    
    # Prepare reference directory if it doesn't exist
    if not os.path.exists(args.reference_dir):
        print(f"Reference directory not found: {args.reference_dir}")
        print("Creating reference images from validation set...")
        args.reference_dir = prepare_reference_images(
            data_root=args.data_root,
            reference_dir=args.reference_dir,
            num_samples=None
        )
    if not os.path.isdir(args.reference_dir):
        raise ValueError(f"Invalid reference directory: {args.reference_dir}")
    if not any(Path(args.reference_dir).glob("*.png")):
        raise RuntimeError("Reference directory contains no images")
    
    print(f"\n{'='*60}")
    print(f"Computing FID")
    print(f"{'='*60}")
    print(f"Generated: {args.generated_dir}")
    print(f"Reference: {args.reference_dir}")
    
    # Compute FID
    fid_score = compute_fid(
        args.generated_dir,
        args.reference_dir,
        args.device,
        args.batch_size
    )
    
    print(f"\n{'='*60}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"{'='*60}")
    
    # Save results to JSON
    results = {
        "fid_score": float(fid_score),
        "generated_dir": args.generated_dir,
        "reference_dir": args.reference_dir
    }
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
