#!/usr/bin/env python3
"""
Template for implementing custom generative models
Students should create their own implementation by inheriting from the base classes.

This file provides skeleton code for implementing generative models.
Students need to implement the TODO sections in their own files.
"""

import torch
import torch.nn.functional as F
from src.base_model import BaseScheduler, BaseGenerativeModel
from src.network import UNet
from torch.utils.checkpoint import checkpoint as _checkpoint


# ============================================================================
# GENERATIVE MODEL SKELETON
# ============================================================================

class CustomScheduler(BaseScheduler):
    """
    Custom Scheduler Skeleton
    
    TODO: Students need to implement this class in their own file.
    Required methods:
    1. sample_timesteps: Sample random timesteps for training
    2. forward_process: Apply forward process to transform data
    3. reverse_process_step: Perform one step of the reverse process
    4. get_target: Get target for model prediction
    """
    
    def __init__(self, num_train_timesteps: int = 1000, **kwargs):
        super().__init__(num_train_timesteps, **kwargs)
        # Rectified Flow uses continuous time t in [0, 1]
        # No discrete schedule parameters are required.
    
    def sample_timesteps(self, batch_size: int, device: torch.device):
        """
        Sample random timesteps for training.
        
        Returns:
            Tensor of shape (batch_size,) with timestep values
        """
        # Uniform sampling over t in [0, 1]
        return torch.rand(batch_size, device=device)
    
    def forward_process(self, data, noise, t):
        """
        Apply forward process to add noise to clean data.
        
        Args:
            data: Clean data tensor
            noise: Noise tensor
            t: Timestep tensor
            
        Returns:
            Noisy data at timestep t
        """
        # Straight-line interpolation: x_t = (1 - t) * x0 + t * eps
        # t shape: (B,). Broadcast to BCHW
        while t.dim() < data.dim():
            t = t.view(-1, *([1] * (data.dim() - 1)))
        return (1.0 - t) * data + t * noise
    
    def reverse_process_step(self, xt, pred, t, t_next):
        """
        Perform one step of the reverse (denoising) process.
        
        Args:
            xt: Current noisy data
            pred: Model prediction (e.g., predicted noise, velocity, or x0)
            t: Current timestep
            t_next: Next timestep
            
        Returns:
            Updated data at timestep t_next
        """
        # Euler step along learned velocity field v_theta
        # x_{t_next} = x_t - (t - t_next) * v_theta(x_t, t)
        dt = (t - t_next)
        while dt.dim() < xt.dim():
            dt = dt.view(-1, *([1] * (xt.dim() - 1)))
        return xt - dt * pred
    
    def get_target(self, data, noise, t):
        """
        Get the target for model prediction (what the network should learn to predict).
        
        Args:
            data: Clean data
            noise: Noise
            t: Timestep
            
        Returns:
            Target tensor (e.g., noise for DDPM, velocity for Flow Matching)
        """
        # Target velocity on straight path: v* = eps - x0
        return noise - data


class CustomGenerativeModel(BaseGenerativeModel):
    """
    Custom Generative Model Skeleton
    
    Students need to implement this class by inheriting from BaseGenerativeModel.
    This class wraps the network and scheduler to provide training and sampling interfaces.
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
        # Optional LPIPS support (enabled via kwargs in train.py if desired)
        self.use_lpips = bool(kwargs.get("use_lpips", False))
        self.lpips_weight = float(kwargs.get("lpips_weight", 0.0))
        self._lpips_net = None
        self.use_checkpoint = bool(kwargs.get("use_checkpoint", False))
        if self.use_lpips and self.lpips_weight > 0:
            try:
                import lpips  # type: ignore
                self._lpips_net = lpips.LPIPS(net='vgg')
            except Exception:
                # If LPIPS is unavailable, silently disable it
                self.use_lpips = False
                self.lpips_weight = 0.0
    
    def compute_loss(self, data, noise, **kwargs):
        """
        Compute the training loss.
        
        Args:
            data: Clean data batch
            noise: Noise batch (or x0 for flow models)
            **kwargs: Additional arguments
            
        Returns:
            Loss tensor
        """
        B = data.shape[0]
        device = data.device
        # Sample time and noise
        t = self.scheduler.sample_timesteps(B, device)
        eps = torch.randn_like(data)
        # Forward process and velocity target
        xt = self.scheduler.forward_process(data, eps, t)
        target_v = self.scheduler.get_target(data, eps, t)
        # Predict velocity
        pred_v = self.predict(xt, t)
        vel_loss = F.mse_loss(pred_v, target_v)

        # Optional LPIPS on reconstructed x0
        total_loss = vel_loss
        if self.use_lpips and self._lpips_net is not None and self.lpips_weight > 0:
            if next(self._lpips_net.parameters()).device != device:
                self._lpips_net = self._lpips_net.to(device)
                self._lpips_net.eval()
            with torch.no_grad():
                pass  # Ensure net is eval; gradients not needed for LPIPS network
            # Reconstruct x0_hat = x_t - t * v_theta(x_t, t)
            t_broadcast = t.view(-1, 1, 1, 1)
            x0_hat = xt - t_broadcast * pred_v
            # LPIPS expects [-1,1] range; data already in [-1,1]
            lp = self._lpips_net(x0_hat, data)
            lp_loss = lp.mean()
            total_loss = total_loss + self.lpips_weight * lp_loss

        return total_loss
    
    def predict(self, xt, t, **kwargs):
        """
        Make prediction given noisy data and timestep.
        
        Args:
            xt: Noisy data
            t: Timestep
            **kwargs: Additional arguments (e.g., condition for additional timestep)
            
        Returns:
            Model prediction
        """
        # Network predicts velocity v_theta(xt, t)
        condition = kwargs.get('condition', None)
        if self.use_checkpoint and xt.requires_grad:
            return _checkpoint(lambda a, b, c: self.network(a, b, c), xt, t, condition)
        return self.network(xt, t, condition)
    
    def sample(self, shape, num_inference_timesteps=20, return_traj=False, verbose=False, **kwargs):
        """
        Generate samples from noise using the reverse process.
        
        Args:
            shape: Shape of samples to generate (batch_size, channels, height, width)
            num_inference_timesteps: Number of denoising steps (NFE)
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            **kwargs: Additional arguments
            
        Returns:
            Generated samples (or trajectory if return_traj=True)
        """
        device = self.device
        B = shape[0]
        x = torch.randn(shape, device=device)
        traj = [x.clone()] if return_traj else None

        def heun_step(x_in, t_i, t_j):
            # One Heun step across [t_i -> t_j]
            t_i_tensor = torch.full((B,), float(t_i), device=device)
            t_j_tensor = torch.full((B,), float(t_j), device=device)
            k1 = self.predict(x_in, t_i_tensor)
            x_euler = self.scheduler.reverse_process_step(x_in, k1, t_i_tensor, t_j_tensor)
            k2 = self.predict(x_euler, t_j_tensor)
            dt = (t_i - t_j)
            x_in.sub_(0.5 * dt * (k1 + k2))
            return x_in

        if num_inference_timesteps == 1:
            # Single Euler step 1 -> 0
            t_i = torch.ones(B, device=device)
            t_j = torch.zeros(B, device=device)
            k1 = self.predict(x, t_i)
            x = self.scheduler.reverse_process_step(x, k1, t_i, t_j)
            if return_traj:
                traj.append(x.clone())
        elif num_inference_timesteps == 2:
            # One Heun step across full interval (2 evaluations)
            x = heun_step(x, 1.0, 0.0)
            if return_traj:
                traj.append(x.clone())
        elif num_inference_timesteps == 4:
            # Two Heun steps (4 evaluations): [1.0->0.5], [0.5->0.0]
            x = heun_step(x, 1.0, 0.5)
            if return_traj:
                traj.append(x.clone())
            x = heun_step(x, 0.5, 0.0)
            if return_traj:
                traj.append(x.clone())
        else:
            # Default: Euler with uniform grid matching NFE
            t_grid = torch.linspace(1.0, 0.0, steps=num_inference_timesteps + 1, device=device)
            for i in range(num_inference_timesteps):
                t_i = torch.full((B,), float(t_grid[i].item()), device=device)
                t_j = torch.full((B,), float(t_grid[i+1].item()), device=device)
                k1 = self.predict(x, t_i)
                x = self.scheduler.reverse_process_step(x, k1, t_i, t_j)
                if return_traj:
                    traj.append(x.clone())

        x = x.clamp(-1.0, 1.0)
        return traj if return_traj else x


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def create_custom_model(device="cpu", **kwargs):
    """
    Example function to create a custom generative model.
    
    Students should modify this function to create their specific model.
    
    Args:
        device: Device to place model on
        **kwargs: Additional arguments that can be passed to network or scheduler
                  (e.g., num_train_timesteps, use_additional_condition for scalar conditions
                   like step size in Shortcut Models or end timestep in Consistency Trajectory Models, etc.)
    """
    
    # Create U-Net backbone with FIXED hyperparameters
    # DO NOT MODIFY THESE HYPERPARAMETERS
    use_additional_condition = kwargs.get('use_additional_condition', False)
    network = UNet(
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_additional_condition=use_additional_condition
    )
    
    # Extract scheduler parameters with defaults
    scheduler_kwargs = dict(kwargs)
    num_train_timesteps = scheduler_kwargs.pop('num_train_timesteps', 1000)
    
    # Create your scheduler
    scheduler = CustomScheduler(num_train_timesteps=num_train_timesteps, **scheduler_kwargs)
    
    # Create your model
    compile_flag = bool(kwargs.get('compile', False))
    if compile_flag:
        try:
            network = torch.compile(network, backend="inductor", mode="max-autotune", fullgraph=True)
        except Exception:
            pass
    model = CustomGenerativeModel(network, scheduler, **kwargs)
    
    return model.to(device)
