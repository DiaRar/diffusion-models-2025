#!/usr/bin/env python3
"""
Skeleton training script for generative models
Students need to implement their own model classes and modify this script accordingly.
"""

import json
import argparse
import torch
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time
import warnings
from tqdm import tqdm
import torch.distributed as dist

from src.network import UNet
from dataset import SimpsonsDataModule, get_data_iterator
from src.utils import tensor_to_pil_image, get_current_time, save_model, seed_everything
from src.base_model import BaseScheduler, BaseGenerativeModel
from custom_model import create_custom_model

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


def train_model(
    model,
    train_iterator,
    num_iterations=100000,
    lr=1e-4,
    save_dir="./results",
    device="cpu",
    log_interval=500,
    save_interval=10000,
    model_config=None,
    use_ema=False,
    ema_decay=0.999,
    plot_interval=None,
    xla_bf16=False,
):
    """
    Train a generative model.
    
    Args:
        model: Generative model to train
        train_iterator: Training data iterator
        num_iterations: Number of training iterations
        lr: Learning rate
        save_dir: Directory to save checkpoints
        device: Device to run training on
        log_interval: Interval for logging
        save_interval: Interval for saving checkpoints and samples
        model_config: Model configuration dictionary to save with checkpoints
    """
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Create optimizer
    if str(device) == "cuda":
        try:
            optimizer = optim.AdamW(model.network.parameters(), lr=lr, fused=True)
        except Exception:
            optimizer = optim.AdamW(model.network.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.network.parameters(), lr=lr)
    try:
        from torch.amp import GradScaler as _GradScaler
        scaler = _GradScaler('cuda', enabled=str(device) == "cuda")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=str(device) == "cuda")
    ema_enabled = bool(use_ema)
    ema_params = None
    if ema_enabled:
        ema_params = [p.detach().clone() for p in model.network.parameters()]
    
    # Save training configuration
    config = {
        'num_iterations': num_iterations,
        'lr': lr,
        'log_interval': log_interval,
        'save_interval': save_interval,
        'device': str(device),
    }
    with open(save_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to: {save_dir / 'training_config.json'}")
    
    # Training loop
    train_losses = []
    # Track best loss (rolling average) for saving a compliant best checkpoint
    best_metric = float("inf")
    best_iter = 0
    checkpoints_dir = Path("./checkpoints")
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Starting training for {num_iterations} iterations...")
    print(f"Model: {type(model).__name__}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    
    model.train()
    
    # Determine master process for XLA to avoid duplicated I/O
    is_xla = (getattr(device, 'type', '') == 'xla') or ('xla' in str(device))
    is_cuda = str(device) == "cuda"
    is_master = True
    if is_xla:
        try:
            import torch_xla.core.xla_model as xm
            is_master = xm.is_master_ordinal()
        except Exception:
            is_master = True
    pbar = tqdm(range(num_iterations), desc="Training") if is_master else range(num_iterations)
    
    for iteration in pbar:
        # Get batch from infinite iterator
        data = next(train_iterator)
        data = data.contiguous(memory_format=torch.channels_last).to(device, non_blocking=True)
        if is_xla and xla_bf16:
            try:
                data = data.to(torch.bfloat16)
            except Exception:
                pass
        try:
            if is_xla:
                loss = model.compute_loss(data, None)
            else:
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    loss = model.compute_loss(data, None)
        except NotImplementedError:
            print("Error: compute_loss method not implemented!")
            print("Please implement the compute_loss method in your model class.")
            return
        except Exception as e:
            print(f"Error computing loss: {e}")
            return
        
        optimizer.zero_grad(set_to_none=True)
        if is_xla:
            import torch_xla.core.xla_model as xm
            import torch_xla as tx
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            tx.sync()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if ema_enabled:
            if is_xla:
                for ema_p, p in zip(ema_params, model.network.parameters()):
                    ema_p.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
            else:
                try:
                    torch._foreach_mul_(ema_params, ema_decay)
                    torch._foreach_add_(ema_params, [p.detach() for p in model.network.parameters()], alpha=1 - ema_decay)
                except Exception:
                    for ema_p, p in zip(ema_params, model.network.parameters()):
                        ema_p.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
        
        train_losses.append(loss.item())
        
        # Update progress bar
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Logging and save loss curve (only master for XLA)
        if is_master and (iteration + 1) % log_interval == 0:
            avg_loss = sum(train_losses[-log_interval:]) / min(log_interval, len(train_losses))
            print(f"Iteration {iteration+1}/{num_iterations}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

            if avg_loss < best_metric:
                best_metric = avg_loss
                best_iter = iteration + 1
                best_path = checkpoints_dir / "best_model.pt"
                try:
                    if ema_enabled:
                        original = [p.detach().clone() for p in model.network.parameters()]
                        for p, ema_p in zip(model.network.parameters(), ema_params):
                            p.data.copy_(ema_p)
                        save_model(model, str(best_path), model_config)
                        for p, orig in zip(model.network.parameters(), original):
                            p.data.copy_(orig)
                    else:
                        save_model(model, str(best_path), model_config)
                    print(f"  New best avg loss {best_metric:.4f} at iter {best_iter}. Saved: {best_path}")
                    print(f"  Config saved: {best_path.parent / 'model_config.json'}")
                except Exception as e:
                    print(f"Warning: Failed to save best model: {e}")
            
            # Save training loss curve at configured interval (defaults to save_interval)
            if plot_interval is None:
                plot_iv = save_interval
            else:
                plot_iv = plot_interval
            if (iteration + 1) % plot_iv == 0:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.plot(train_losses)
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.title('Training Loss')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Warning: Could not save training curve: {e}")
        
        if is_master and (iteration + 1) % save_interval == 0:
            checkpoint_path = save_dir / f"checkpoint_iter_{iteration+1}.pt"
            try:
                if ema_enabled:
                    original = [p.detach().clone() for p in model.network.parameters()]
                    for p, ema_p in zip(model.network.parameters(), ema_params):
                        p.data.copy_(ema_p)
                    save_model(model, str(checkpoint_path), model_config)
                    last_ckpt = save_dir / "last.ckpt"
                    save_model(model, str(last_ckpt), model_config)
                    for p, orig in zip(model.network.parameters(), original):
                        p.data.copy_(orig)
                else:
                    save_model(model, str(checkpoint_path), model_config)
                    last_ckpt = save_dir / "last.ckpt"
                    save_model(model, str(last_ckpt), model_config)
                print(f"\n  Checkpoint saved: {checkpoint_path}")
                print(f"  Last checkpoint saved: {last_ckpt}")
                print(f"  Config saved: {checkpoint_path.parent / 'model_config.json'}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
        
            print("\n  Generating samples...")
            model.eval()
            shape = (4, 3, 64, 64)
            with torch.no_grad():
                samples = model.sample(
                    shape,
                    num_inference_timesteps=20
                )
                if is_xla and xla_bf16:
                    try:
                        samples = samples.to(torch.float32)
                    except Exception:
                        pass
            model.train()
            
            # Save samples
            pil_images = tensor_to_pil_image(samples)
            for i, img in enumerate(pil_images):
                img.save(save_dir / f"iter={iteration+1}_sample_{i}.png")
        
    # Save final model
    final_path = save_dir / "final_model.pt"
    # Ensure a compliant best checkpoint exists even if no log interval triggered (master only)
    try:
        if is_master and best_metric == float("inf") and len(train_losses) > 0:
            overall_avg = sum(train_losses) / len(train_losses)
            best_metric = overall_avg
            best_iter = len(train_losses)
            best_path = Path("./checkpoints") / "best_model.pt"
            if ema_enabled:
                original = [p.detach().clone() for p in model.network.parameters()]
                for p, ema_p in zip(model.network.parameters(), ema_params):
                    p.data.copy_(ema_p)
                save_model(model, str(best_path), model_config)
                for p, orig in zip(model.network.parameters(), original):
                    p.data.copy_(orig)
            else:
                save_model(model, str(best_path), model_config)
            print(f"Saved best model at end (avg {best_metric:.4f}, iter {best_iter}): {best_path}")
            print(f"Config saved: {best_path.parent / 'model_config.json'}")
    except Exception as e:
        print(f"Warning: Could not save fallback best model: {e}")
    try:
        if is_master and ema_enabled:
            original = [p.detach().clone() for p in model.network.parameters()]
            for p, ema_p in zip(model.network.parameters(), ema_params):
                p.data.copy_(ema_p)
            save_model(model, str(final_path), model_config)
            last_ckpt = Path(save_dir) / "last.ckpt"
            save_model(model, str(last_ckpt), model_config)
            for p, orig in zip(model.network.parameters(), original):
                p.data.copy_(orig)
        elif is_master:
            save_model(model, str(final_path), model_config)
            last_ckpt = Path(save_dir) / "last.ckpt"
            save_model(model, str(last_ckpt), model_config)
        if is_master:
            print(f"Final model saved: {final_path}")
            print(f"Last checkpoint saved: {last_ckpt}")
            print(f"Config saved: {final_path.parent / 'model_config.json'}")
    except Exception as e:
        print(f"Error saving final model: {e}")
    
    print(f"\nTraining completed! Results saved to: {save_dir}")
    print("Check the training_curves.png for loss visualization.")

def tpu_worker_entry(index, args):
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed import parallel_loader as pl
    device = xm.xla_device()
    is_master = xm.is_master_ordinal()

    seed_everything(args.seed + index)
    if is_master:
        print(f"[TPU worker {index}] Device: {device}")

    excluded_keys = ['device', 'batch_size', 'num_iterations', 'lr', 'save_dir', 'log_interval', 'save_interval', 'seed']
    model_kwargs = {}
    for key, value in args.__dict__.items():
        if key not in excluded_keys and value is not None:
            model_kwargs[key] = value

    model = create_custom_model(device=device, **model_kwargs)
    model_config = {
        'model_type': type(model).__name__,
        'scheduler_type': type(model.scheduler).__name__,
        **model_kwargs
    }

    data_module = SimpsonsDataModule(
        batch_size=args.batch_size,
        num_workers=(args.num_workers if args.num_workers is not None else 4),
    )
    if bool(getattr(args, 'cache_dataset', False)):
        try:
            data_module.train_ds = CachedDataset(data_module.train_ds, pin_memory=False)
        except Exception:
            pass
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    sampler = torch.utils.data.distributed.DistributedSampler(
        data_module.train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    loader_kwargs = {
        "pin_memory": False,
        "persistent_workers": bool(getattr(args, 'persistent_workers', True)),
        "prefetch_factor": int(getattr(args, 'prefetch_factor', 4)),
        "shuffle": False,
        "drop_last": True,
        "sampler": sampler,
    }
    train_loader = torch.utils.data.DataLoader(
        data_module.train_ds,
        batch_size=args.batch_size,
        num_workers=(args.num_workers if args.num_workers is not None else 4),
        **loader_kwargs,
    )
    mp_loader = pl.MpDeviceLoader(train_loader, device)
    train_iterator = get_data_iterator(mp_loader)

    train_model(
        model=model,
        train_iterator=train_iterator,
        num_iterations=args.num_iterations,
        lr=args.lr,
        save_dir=args.save_dir,
        device=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        model_config=model_config,
        use_ema=bool(getattr(args, 'use_ema', False)),
        ema_decay=float(getattr(args, 'ema_decay', 0.999)),
        plot_interval=getattr(args, 'plot_interval', None),
        xla_bf16=bool(getattr(args, 'xla_bf16', False)),
    )


def main(args):
    # Set seed for reproducibility
    seed_everything(args.seed)
    print(f"Seed set to: {args.seed}")
    
    # Add timestamp if save directory is not specified
    if args.save_dir == "./results":
        args.save_dir = f"./results/{get_current_time()}"
    
    # Reduce noisy warnings from third-party libraries
    try:
        warnings.filterwarnings("ignore", message=".*invalid escape sequence.*", category=SyntaxWarning)
    except Exception:
        pass
    # Set device (TPU-first logic to avoid redundant CPU print)
    use_tpu = (str(args.device).lower() == "tpu")
    use_ddp = bool(getattr(args, 'ddp', False)) and (str(args.device).lower() == 'cuda')
    local_rank = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', 0)))
    if use_tpu:
        if bool(getattr(args, 'xla_spawn', False)):
            try:
                if bool(getattr(args, 'xla_profile', False)):
                    os.environ.setdefault('XLA_FLAGS', '--xla_hlo_profile')
            except Exception:
                pass
            device = 'xla'
            print(f"Using XLA device: {device}")
        else:
            try:
                if bool(getattr(args, 'xla_profile', False)):
                    os.environ.setdefault('XLA_FLAGS', '--xla_hlo_profile')
                use_mp_loader = bool(getattr(args, 'xla_use_mp_device_loader', True))
                import torch_xla as tx
                device = tx.device()
            except Exception:
                try:
                    import torch_xla.core.xla_model as xm
                    device = xm.xla_device()
                except Exception as e:
                    print(f"TPU requested but torch_xla is unavailable: {e}")
                    return
            print(f"Using XLA device: {device}")
    else:
        if use_ddp and torch.cuda.is_available():
            if not dist.is_initialized():
                backend = getattr(args, 'dist_backend', 'nccl')
                dist.init_process_group(backend=backend)
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            if dist.get_rank() == 0:
                print(f"Using device: {device}")
        else:
            device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
            print(f"Using device: {device}")
        try:
            if str(device) == "cuda":
                import torch.backends.cuda as cuda_backends
                cuda_backends.matmul.allow_tf32 = True
                import torch.backends.cudnn as cudnn
                cudnn.allow_tf32 = True
                cudnn.benchmark = True
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    
    # XLA multi-core spawn (aggressive TPU data-parallel)
    if use_tpu and bool(getattr(args, 'xla_spawn', False)):
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.xla_multiprocessing as xmp
        except Exception as e:
            print(f"XLA spawn requested but torch_xla is unavailable: {e}")
            return

        def _tpu_worker(index, args):
            # Device
            import torch_xla.core.xla_model as xm
            from torch_xla.distributed import parallel_loader as pl
            device = xm.xla_device()
            is_master = xm.is_master_ordinal()

            # Seed
            seed_everything(args.seed + index)
            if is_master:
                print(f"[TPU worker {index}] Device: {device}")

            # Build model kwargs
            excluded_keys = ['device', 'batch_size', 'num_iterations', 'lr', 'save_dir', 'log_interval', 'save_interval', 'seed']
            model_kwargs = {}
            for key, value in args.__dict__.items():
                if key not in excluded_keys and value is not None:
                    model_kwargs[key] = value

            # Create model
            model = create_custom_model(device=device, **model_kwargs)
            model_config = {
                'model_type': type(model).__name__,
                'scheduler_type': type(model.scheduler).__name__,
                **model_kwargs
            }

            # Dataset and distributed sampler
            data_module = SimpsonsDataModule(
                batch_size=args.batch_size,
                num_workers=(args.num_workers if args.num_workers is not None else 4),
            )
            if bool(getattr(args, 'cache_dataset', False)):
                try:
                    data_module.train_ds = CachedDataset(data_module.train_ds, pin_memory=False)
                except Exception:
                    pass
            world_size = xm.xrt_world_size()
            rank = xm.get_ordinal()
            sampler = torch.utils.data.distributed.DistributedSampler(
                data_module.train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
            )
            loader_kwargs = {
                "pin_memory": False,
                "persistent_workers": bool(getattr(args, 'persistent_workers', True)),
                "prefetch_factor": int(getattr(args, 'prefetch_factor', 4)),
                "shuffle": False,
                "drop_last": True,
                "sampler": sampler,
            }
            train_loader = torch.utils.data.DataLoader(
                data_module.train_ds,
                batch_size=args.batch_size,
                num_workers=(args.num_workers if args.num_workers is not None else 4),
                **loader_kwargs,
            )
            mp_loader = pl.MpDeviceLoader(train_loader, device)
            train_iterator = get_data_iterator(mp_loader)

            # Train
            train_model(
                model=model,
                train_iterator=train_iterator,
                num_iterations=args.num_iterations,
                lr=args.lr,
                save_dir=args.save_dir,
                device=device,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                model_config=model_config,
                use_ema=bool(getattr(args, 'use_ema', False)),
                ema_decay=float(getattr(args, 'ema_decay', 0.999)),
                plot_interval=getattr(args, 'plot_interval', None),
                xla_bf16=bool(getattr(args, 'xla_bf16', False)),
            )

        nprocs_env = str(getattr(args, 'xla_num_cores', 8))
        try:
            os.environ.setdefault('TPU_NUM_DEVICES', nprocs_env)
        except Exception:
            pass
        xmp.spawn(tpu_worker_entry, args=(args,), nprocs=None, start_method='fork')
        return

    # Create model
    print("Creating model...")

    # Prepare kwargs for create_custom_model from args
    model_kwargs = {}
    # Pass all custom arguments except training-specific ones
    # Network hyperparameters (ch, ch_mult, attn, num_res_blocks, dropout) are FIXED
    # Students can add their own custom arguments for scheduler/model configuration
    excluded_keys = ['device', 'batch_size', 'num_iterations', 
                     'lr', 'save_dir', 'log_interval', 'save_interval', 'seed']
    for key, value in args.__dict__.items():
        if key not in excluded_keys and value is not None:
            model_kwargs[key] = value

    try:
        model = create_custom_model(
            device=device,
            **model_kwargs
        )
        print(f"Model created: {type(model).__name__}")
        print(f"Scheduler: {type(model.scheduler).__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.network.parameters()):,}")
    except NotImplementedError as e:
        print(f"Error: {e}")
        print("Please implement the CustomScheduler and CustomGenerativeModel classes in custom_model.py")
        return
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Prepare model configuration for reproducibility
    # This will be saved together with each checkpoint
    model_config = {
        'model_type': type(model).__name__,
        'scheduler_type': type(model.scheduler).__name__,
        **model_kwargs  # Include all custom model arguments
    }
    
    # Optional channels_last for model on CUDA
    try:
        if str(device) == "cuda" and bool(getattr(args, 'channels_last', True)):
            model.network = model.network.to(memory_format=torch.channels_last)
            for p in model.network.parameters():
                p.data = p.data.contiguous(memory_format=torch.channels_last)
    except Exception:
        pass

    # TPU: keep model in float32 for compatibility with provided TimeEmbedding

    # Load dataset
    print("Loading dataset...")
    try:
        data_module = SimpsonsDataModule(
            batch_size=args.batch_size,
            num_workers=(args.num_workers if args.num_workers is not None else 4),
        )
        if bool(getattr(args, 'cache_dataset', False)):
            try:
                data_module.train_ds = CachedDataset(
                    data_module.train_ds,
                    pin_memory=(False if use_tpu else bool(getattr(args, 'pin_memory', True)))
                )
            except Exception:
                pass
        
        base_loader = data_module.train_dataloader()
        try:
            sampler = None
            if use_ddp:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    data_module.train_ds,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                    drop_last=True,
                )
            loader_kwargs = {
                "pin_memory": (False if use_tpu else bool(getattr(args, 'pin_memory', True))),
                "persistent_workers": bool(getattr(args, 'persistent_workers', True)),
                "prefetch_factor": int(getattr(args, 'prefetch_factor', 4)),
                "shuffle": (False if sampler is not None else True),
                "drop_last": True,
                "sampler": sampler,
            }
            train_loader = torch.utils.data.DataLoader(
                data_module.train_ds,
                batch_size=args.batch_size,
                num_workers=(args.num_workers if args.num_workers is not None else 4),
                **loader_kwargs,
            )
        except Exception:
            train_loader = base_loader
        if use_tpu:
            try:
                # Prefer modern MpDeviceLoader for XLA
                from torch_xla.distributed import parallel_loader as pl
                mp_loader = pl.MpDeviceLoader(train_loader, device)
                train_iterator = get_data_iterator(mp_loader)
            except Exception:
                import torch_xla.distributed.parallel_loader as pl
                ploader = pl.ParallelLoader(train_loader, [device])
                per_dev_loader = ploader.per_device_loader(device)
                train_iterator = get_data_iterator(per_dev_loader)
        else:
            train_iterator = get_data_iterator(train_loader)
        
        print(f"Total iterations: {args.num_iterations}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Train model
    # Determine master for logging/saving
    if use_ddp:
        is_master = (dist.get_rank() == 0)
    else:
        is_master = True

    train_model(
        model=model,
        train_iterator=train_iterator,
        num_iterations=args.num_iterations,
        lr=args.lr,
        save_dir=args.save_dir,
        device=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        model_config=model_config,
        use_ema=bool(getattr(args, 'use_ema', False)),
        ema_decay=float(getattr(args, 'ema_decay', 0.999)),
        plot_interval=getattr(args, 'plot_interval', None),
        xla_bf16=bool(getattr(args, 'xla_bf16', False)),
    )

    # Cleanup DDP
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generative model")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--num_iterations", type=int, default=100000,
                       help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./results",
                       help="Directory to save")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run training on")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Interval for logging")
    parser.add_argument("--save_interval", type=int, default=10000,
                       help="Interval for saving checkpoints and samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    # Model-specific arguments (students can add more)
    # DO NOT MODIFY THE PROVIDED NETWORK HYPERPARAMETERS 
    # (ch=128, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=4, dropout=0.1)
    parser.add_argument("--use_additional_condition", action="store_true",
                       help="Use additional condition embedding in U-Net (e.g., step size for Shortcut Models or end timestep for Consistency Trajectory Models)")
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                       help="Number of training timesteps for scheduler")
    
    # Optional perceptual loss
    parser.add_argument("--use_lpips", action="store_true",
                       help="Enable LPIPS perceptual loss on reconstructed x0 (default: disabled)")
    parser.add_argument("--lpips_weight", type=float, default=0.0,
                       help="Weight for LPIPS loss term (default: 0.0)")
    parser.add_argument("--use_ema", action="store_true",
                       help="Enable EMA for model parameters")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                       help="EMA decay factor")
    # Performance-related toggles
    parser.add_argument("--compile", action="store_true", default=True,
                       help="Compile model with torch.compile for speed")
    parser.add_argument("--channels_last", action="store_true", default=True,
                       help="Use channels_last memory format on CUDA")
    parser.add_argument("--compile_mode", type=str, choices=["default","reduce-overhead","max-autotune"], default=None,
                       help="torch.compile mode for CUDA GPUs (auto if omitted)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of DataLoader workers (default: CPU count)")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                       help="DataLoader prefetch factor (items per worker)")
    parser.add_argument("--persistent_workers", action="store_true",
                       help="Keep DataLoader workers alive for faster epochs")
    parser.add_argument("--pin_memory", action="store_true",
                       help="Pin host memory for faster H2D copies on CUDA")
    parser.add_argument("--cache_dataset", action="store_true",
                       help="Cache and pre-transform dataset in memory for speed")
    parser.add_argument("--plot_interval", type=int, default=None,
                       help="Iterations between plotting loss curve (default: save_interval)")
    # TPU/XLA performance flags
    parser.add_argument("--xla_bf16", action="store_true",
                       help="Use bfloat16 training on XLA for speed")
    parser.add_argument("--xla_use_mp_device_loader", action="store_true",
                       help="Use MpDeviceLoader for TPU dataloading")
    parser.add_argument("--xla_profile", action="store_true",
                       help="Enable XLA HLO performance profiling")
    parser.add_argument("--xla_spawn", action="store_true",
                       help="Spawn data-parallel training across TPU cores")
    parser.add_argument("--xla_num_cores", type=int, default=8,
                       help="Number of TPU cores to spawn")
    
    # Students can add their own custom arguments below for their implementation
    # For example:
    # parser.add_argument("--sigma_min", type=float, default=0.001, help="Minimum noise level")
    # parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    # etc.
    
    args = parser.parse_args()
    # Defaults for performance flags if not explicitly set
    if args.num_workers is None:
        try:
            args.num_workers = max(4, os.cpu_count() or 4)
        except Exception:
            args.num_workers = 4

    main(args)
