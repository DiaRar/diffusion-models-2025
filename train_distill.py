#!/usr/bin/env python3
"""
NEW FILE: Distillation training script for Latent Consistency Model (LCM).

This script:
1. Loads the "teacher" model (the Rectified Flow model trained by train.py).
2. Creates a "student" LCM model (using the *same* fixed UNet architecture).
3. Trains the student model using the Multistep Latent Consistency Distillation loss.
4. Saves the final distilled (student) model, which is capable of 1-4 step inference.
"""

import json
import argparse
import torch
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import time
import warnings
from tqdm import tqdm

from dataset import SimpsonsDataModule, get_data_iterator
from src.utils import tensor_to_pil_image, get_current_time, save_model, seed_everything, load_model
from custom_model import create_custom_model as create_teacher_model
from custom_model_lcm import create_lcm_model as create_student_model

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, pin_memory=False):
        self.base_ds = base_ds
        self.pin_memory = bool(pin_memory)
        cached = []
        for i in range(len(base_ds)):
            item = base_ds[i]
            if self.pin_memory and isinstance(item, torch.Tensor):
                item = item.pin_memory()
            cached.append(item)
        self._cache = cached

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        return self._cache[idx]


def train_distill(
    teacher_model,
    student_model,
    train_iterator,
    num_iterations=50000, # Distillation is often faster
    lr=1e-4,
    save_dir="./results_distill",
    device="cpu",
    log_interval=200,
    save_interval=5000,
    plot_interval=5000,
    model_config=None,
    num_distill_steps=4, # Number of steps for the student (e.g., 4 for NFE=4)
    grad_accum_steps=1,
    use_ema=False,
    ema_decay=0.999,
    **kwargs
):
    """
    Main distillation training loop
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Freeze the teacher model
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print("✓ Teacher model loaded and frozen.")

    # Create optimizer for the STUDENT model
    optimizer = optim.AdamW(student_model.parameters(), lr=lr)

    # Prepare EMA for the STUDENT model
    ema_model = None
    if use_ema:
        from copy import deepcopy
        ema_model = deepcopy(student_model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False
        print("✓ EMA enabled for student model.")

    # Checkpoint paths
    final_ckpt_path = Path("./checkpoints/best_lcm_model.pt") # This is the final model
    final_cfg_path = Path("./checkpoints/model_config_lcm.json")
    final_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    last_ckpt_path = save_dir / "last_lcm.ckpt"
    if model_config is None:
        model_config = {}
    
    # Save initial config
    save_model(student_model, final_ckpt_path, model_config, final_cfg_path)
    
    print(f"Starting distillation for {num_iterations} iterations...")
    print(f"  Device: {device}")
    print(f"  Learning Rate: {lr}")
    print(f"  Distillation Steps (K): {num_distill_steps}")
    print(f"  Save Directory: {save_dir}")

    losses = []
    pbar = tqdm(range(num_iterations))
    start_time = time.time()
    
    for i in pbar:
        try:
            student_model.train()
            
            # --- Gradient Accumulation ---
            total_loss_accum = 0.0
            optimizer.zero_grad()
            
            for _ in range(grad_accum_steps):
                batch = next(train_iterator)
                data = batch.to(device)
                
                if model_config.get("channels_last", False):
                    data = data.to(memory_format=torch.channels_last)
                
                # --- Compute Distillation Loss ---
                loss = student_model.compute_loss_distill(
                    data, 
                    teacher_model, 
                    num_distill_steps=num_distill_steps
                )
                
                loss = loss / grad_accum_steps
                loss.backward()
                total_loss_accum += loss.item()

            # --- Optimizer Step ---
            optimizer.step()
            
            losses.append(total_loss_accum)
            pbar.set_description(f"Distill Loss: {total_loss_accum:.4f}")

            # --- EMA Update (Student) ---
            if ema_model is not None:
                for ema_p, p in zip(ema_model.parameters(), student_model.parameters()):
                    ema_p.data.lerp_(p.data, 1.0 - ema_decay)

            # --- Logging ---
            if (i + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Iter {i+1}/{num_iterations} | Distill Loss: {total_loss_accum:.4f} | Time: {elapsed_time:.2f}s")
                start_time = time.time()

            # --- Save Checkpoint & Plot Loss ---
            if (i + 1) % save_interval == 0:
                model_to_save = ema_model if ema_model is not None else student_model
                save_model(model_to_save, last_ckpt_path, model_config)
                print(f"✓ Checkpoint saved to {last_ckpt_path}")
                
                plt.figure(figsize=(10, 5))
                plt.plot(losses)
                plt.title("Distillation Training Loss")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.savefig(save_dir / "distill_training_curves.png")
                plt.close()

            # --- Plot Sample Images (from STUDENT) ---
            if (i + 1) % plot_interval == 0:
                student_model.eval()
                model_to_sample = ema_model if ema_model is not None else student_model
                
                with torch.no_grad():
                    samples_nfe1 = model_to_sample.sample((4, 3, 64, 64), num_inference_timesteps=1)
                    samples_nfe2 = model_to_sample.sample((4, 3, 64, 64), num_inference_timesteps=2)
                    samples_nfe4 = model_to_sample.sample((4, 3, 64, 64), num_inference_timesteps=4)
                
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))
                for j, img in enumerate(tensor_to_pil_image(samples_nfe1)):
                    axes[0, j].imshow(img)
                    axes[0, j].set_title("NFE=1")
                    axes[0, j].axis("off")
                for j, img in enumerate(tensor_to_pil_image(samples_nfe2)):
                    axes[1, j].imshow(img)
                    axes[1, j].set_title("NFE=2")
                    axes[1, j].axis("off")
                for j, img in enumerate(tensor_to_pil_image(samples_nfe4)):
                    axes[2, j].imshow(img)
                    axes[2, j].set_title("NFE=4")
                    axes[2, j].axis("off")
                    
                plt.suptitle(f"Student Model Samples at Iteration {i+1}")
                plt.tight_layout()
                plt.savefig(save_dir / f"samples_iter_{i+1}_NFE_1-2-4.png")
                plt.close()

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving final model...")
            break
        except Exception as e:
            print(f"\nError during training at iteration {i}: {e}")
            break

    print("Distillation finished.")
    
    # Save final model
    model_to_save = ema_model if ema_model is not None else student_model
    save_model(model_to_save, final_ckpt_path, model_config, final_cfg_path)
    print(f"✓ Final LCM (student) model saved to {final_ckpt_path}")
    print(f"✓ Final LCM (student) config saved to {final_cfg_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a Latent Consistency Model (Student)")
    
    # --- Key Args ---
    parser.add_argument("--teacher_ckpt", type=str, default="./checkpoints/best_model.pt",
                       help="Path to the trained teacher model checkpoint")
    parser.add_argument("--num_iterations", type=int, default=50000, 
                       help="Total number of distillation iterations")
    parser.add_argument("--num_distill_steps", type=int, default=4,
                       help="Number of steps for the multistep consistency loss (e.g., 4)")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default=f"./results_distill/{get_current_time()}", 
                       help="Directory to save logs and checkpoints")
    parser.add_argument("--log_interval", type=int, default=200, help="Log loss every N iterations")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save checkpoint every N iterations")
    parser.add_argument("--plot_interval", type=int, default=2500, help="Plot samples every N iterations")

    # Model Config (from teacher config)
    parser.add_argument("--use_ema", action="store_true", help="Use Exponential Moving Average for student")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")

    # Data loading
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for DataLoader")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch factor for DataLoader")
    parser.add_argument("--persistent_workers", action="store_true", help="Use persistent workers")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for DataLoader")
    parser.add_argument("--cache_dataset", action="store_true", help="Cache dataset in memory")
    
    # Grad Accum
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)
    
    device = torch.device(args.device)
    if "xla" in str(device):
        print("TPU/XLA not yet supported by this script. Defaulting to CUDA/CPU.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Load Teacher Config ---
    teacher_config_path = Path(args.teacher_ckpt).parent / "model_config.json"
    if not teacher_config_path.exists():
        raise FileNotFoundError(f"Teacher config not found at {teacher_config_path}")
    
    with open(teacher_config_path, 'r') as f:
        teacher_config = json.load(f)
    print(f"✓ Loaded teacher config from {teacher_config_path}")

    # --- Data Module ---
    dm = SimpsonsDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    if args.cache_dataset:
        print("Caching dataset...")
        dm.train_ds = CachedDataset(dm.train_ds, pin_memory=args.pin_memory)
        print(f"✓ Dataset cached ({len(dm.train_ds)} images)")

    # --- DataLoader ---
    train_loader = torch.utils.data.DataLoader(
        dm.train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor
    )
    train_iterator = get_data_iterator(train_loader)
    
    # --- Load Teacher Model ---
    teacher_model = load_model(
        args.teacher_ckpt, 
        create_teacher_model, # From custom_model.py
        str(device),
        teacher_config_path
    )
    
    # --- Create Student Model ---
    # Inherit config from teacher, but override key LCM flags
    student_config = teacher_config.copy()
    # Remove training/runtime-only keys to avoid duplicate kwargs
    for _k in ['device','batch_size','num_iterations','lr','save_dir','log_interval','save_interval','seed',
               'num_workers','prefetch_factor','persistent_workers','pin_memory','cache_dataset',
               'grad_accum_steps','xla_bf16','xla_use_mp_device_loader','xla_profile','xla_spawn',
               'xla_num_cores','plot_interval']:
        student_config.pop(_k, None)
    student_config.update({
        "model_type": "CustomGenerativeModelLCM",
        "scheduler_type": "LCMScheduler",
        "use_additional_condition": True, # This is the critical flag!
        "num_distill_steps": args.num_distill_steps,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay
    })
    
    student_model = create_student_model(device=device, **student_config)

    # --- Start Training ---
    train_distill(
        teacher_model=teacher_model,
        student_model=student_model,
        train_iterator=train_iterator,
        num_iterations=args.num_iterations,
        lr=args.lr,
        save_dir=args.save_dir,
        device=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        model_config=student_config,
        num_distill_steps=args.num_distill_steps,
        plot_interval=args.plot_interval,
        grad_accum_steps=args.grad_accum_steps,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message="Given `type` is not string")
    warnings.filterwarnings("ignore", category=UserWarning, message="The given NumPy array is not writeable")
    
    main()