#!/usr/bin/env python3
"""
NEW FILE: Implements the "Student" model for Latent Consistency Distillation.

This model is trained by 'train_distill.py' and is the final model to be submitted.
It uses the 'use_additional_condition' flag in the UNet  to enable
few-step generation, as described in the Consistency Model papers.[9]
"""

import torch
import torch.nn.functional as F
from src.base_model import BaseScheduler, BaseGenerativeModel

# Import the *original* create_custom_model function to build the network
# This ensures we use the exact same (unmodified) UNet architecture
from custom_model import create_custom_model as create_base_network_model


class LCMScheduler(BaseScheduler):
    """
    A scheduler for the Latent Consistency Model (LCM).
    The "forward" and "target" methods are not used for the distilled model.
    We only need it as a container and for the reverse step logic.
    """
    def __init__(self, num_train_timesteps: int = 1000, **kwargs):
        super().__init__(num_train_timesteps, **kwargs)
        self.num_train_timesteps = num_train_timesteps
        print("✓ LCM Scheduler initialized.")

    def sample_timesteps(self, batch_size: int, device: torch.device):
        # Sample random discrete steps for distillation
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
    
    def get_target(self, data, noise, t):
        # Not used by the LCM student model
        raise NotImplementedError

    def forward_process(self, data, noise, t):
        # Not used by the LCM student model
        raise NotImplementedError

    def reverse_process_step(self, xt, pred_x0, t_current, t_target):
        """
        Performs one step of the LCM reverse process.
        This is the "consistency" function: f(xt, t) = x0
        We can jump from xt to x_target_t using the predicted x0.
        
        This uses the Rectified Flow path:
        x_target = (1 - t_target) * x0 + t_target * x1
        x_current = (1 - t_current) * x0 + t_current * x1
        
        Solving for x1 and substituting, we get:
        x_target = ( (1 - t_target) / (1 - t_current) ) * x_current + \
                   ( t_target - t_current * (1 - t_target) / (1 - t_current) ) * pred_x0
                   
        (Simplified from the paper for straight paths)
        A simpler 1-step Euler jump using the velocity (v = (x0 - xt) / (1-t)):
        v = (pred_x0 - xt) / (1 - t_current + 1e-8)
        x_target = xt + v * (t_target - t_current)
        """
        t_c = t_current.view(-1, 1, 1, 1).to(xt.dtype)
        t_t = t_target.view(-1, 1, 1, 1).to(xt.dtype)
        
        # Predict velocity: v = (x0 - (1-t)*x_noise_source) / t 
        # On RF path: v = x1 - x0. x0 = xt - t*v. x1 = (xt - (1-t)x0)/t
        # This is complex. Let's use the simplest Euler step, which is what
        # consistency models do for their 1-step generation.
        # The model predicts x0.
        # We can then get the implied velocity: v = (xt - (1-t)*x0) / t
        # But for RF, v = x1 - x0 = (xt - (1-t)x0)/t - x0
        
        # Let's use the "pseudo-Euler" step from LCM:
        # x_prev = x_t + (t_prev - t) * v_pred
        # v_pred = (x_t - (1-t)*x0_pred) / t  -- DDIM-style velocity
        # v_pred_rf = (x0_pred - xt) / (1-t) -- Simpler RF-based velocity
        
        dt = (t_t - t_c) # e.g., (0.0 - 1.0) = -1.0
        
        # Implied velocity from RF path: v = (xt - (1-t)*x0) / t
        # This is for DDPM. For RF, v = x1 - x0.
        # x0_pred = (xt - t*v_pred_rf) / (1-t)
        
        # The student model *predicts* x0.
        # We can use this x0 to get the velocity v = (xt - (1-t)x0) / t
        # Or for RF, the velocity v = x1 - x0. 
        # x1 = (xt - (1-t)x0) / t
        # v = (xt - (1-t)x0)/t - x0 = (xt - x0) / t
        
        # Let's use the simplest velocity: v = pred_x0 - x_noise_source (eps)
        # This is all too complex. The *point* of consistency models
        # is that the model f(x, t) predicts x0.
        # The sampler just calls the model K times.
        
        # The simplest reverse step is just an Euler step with the
        # implied velocity v = (pred_x0 - xt) / (1-t)
        
        v_implied = (pred_x0 - xt) / (1.0 - t_c + 1e-8) # Avoid div by zero at t=1
        x_target = xt + v_implied * dt
        return x_target


class CustomGenerativeModelLCM(BaseGenerativeModel):
    """
    Custom Generative Model - This is the "Student" Model (LCM)
    
    This model is trained via distillation from the teacher.
    It uses the `use_additional_condition=True` flag to accept a
    target timestep `s` and learns to predict x0 in K steps.
    """
    
    def __init__(self, network, scheduler, **kwargs):
        super().__init__(network, scheduler, **kwargs)
        self.num_train_timesteps = scheduler.num_train_timesteps
        print("✓ LCM Student Model initialized (use_additional_condition=True)")

    def compute_loss(self, data, noise, **kwargs):
        """
        This model is not trained with compute_loss.
        It is trained with compute_loss_distill.
        """
        raise NotImplementedError("This is a student model. Run train_distill.py")

    def compute_loss_distill(self, data, teacher_model, num_distill_steps=4):
        """
        Compute the Multistep Latent Consistency Distillation loss.
        """
        B = data.shape[0]
        device = data.device
        
        # 1. Sample noise
        eps = torch.randn_like(data, memory_format=torch.channels_last)
        
        # 2. Sample random timesteps `t_n+1` and `t_n` from the schedule
        # We'll use the discrete K-step schedule for distillation
        t_grid = torch.linspace(1.0, 0.0, num_distill_steps + 1, device=device)
        
        # Sample a random segment index `k` from [0, K-1]
        k = torch.randint(0, num_distill_steps, (B,), device=device)
        
        t_n_plus_1 = t_grid[k].to(data.dtype)     # e.g., t=1.0 (k=0)
        t_n = t_grid[k+1].to(data.dtype)         # e.g., t=0.75 (k=0)
        
        # 3. Get x_t_n+1 using the forward process
        xt_n_plus_1 = self.scheduler.forward_process(data, eps, t_n_plus_1) # RF forward
        xt_n_plus_1 = xt_n_plus_1.contiguous(memory_format=torch.channels_last)

        # 4. Get "ground truth" target from the teacher model
        # We need the teacher's prediction for the step from t_n+1 to t_n
        with torch.no_grad():
            # Use teacher's Heun solver (2 NFE) for a high-quality jump
            # We need to define a simple heun_step function here
            def heun_step(x_in, t_i_val, t_j_val):
                t_i = torch.full((B,), t_i_val, device=device, dtype=x_in.dtype)
                t_j = torch.full((B,), t_j_val, device=device, dtype=x_in.dtype)
                
                k1 = teacher_model.predict(x_in, t_i)
                x_euler = teacher_model.scheduler.reverse_process_step(x_in, k1, t_i, t_j)
                k2 = teacher_model.predict(x_euler, t_j)
                
                dt = (t_i_val - t_j_val)
                x_out = x_in - 0.5 * dt * (k1 + k2)
                return x_out
            
            # Target is the teacher's 1-step solution over the segment
            x_n_target = heun_step(xt_n_plus_1, t_n_plus_1.item(), t_n.item())
            
            # The target for the student's x0 prediction is this x_n_target
            # (Consistency Trajectory Model)
            # Or, for LCM, the target is the x0 predicted by the teacher
            # Let's use the CTM target, as it's more general for multistep
            
            # The student is conditioned on the *target* time t_n
            # and predicts the state x_n
            
            # Re-read MLCM paper [9]: Student predicts f(x_t_n+1, t_n)
            # This is a Consistency Trajectory Model.
            # The UNet `condition` will be the target time `t_n`
            
            # The loss is d( student(x_t_n+1, condition=t_n), target=x_n_target )
            
            # Wait, the student f(x,t) is supposed to predict x0.
            # Let's use the "denoising consistency" from MLCM [9]:
            # Target: x0_pred_teacher = teacher.predict(xt_n_plus_1, t_n_plus_1)
            # This is velocity. We must convert to x0.
            # x0 = (xt - t*v) / (1-t)
            with torch.no_grad():
                heun_step(xt_n_plus_1, t_n_plus_1.item(), 0.0)

        # 5. Get "student" prediction
        # The student is trained to predict x0 from any t.
        # We pass t_n_plus_1 as the main timestep
        # We pass t_n as the *conditional* timestep
        # This trains the "Multistep" (CTM) property
        
        # This is what the student predicts
        x_n_pred_student = self.predict(xt_n_plus_1, t_n_plus_1, condition=t_n)
        
        # The loss is the L2 distance between the student's jump and teacher's jump
        loss = F.mse_loss(x_n_pred_student, x_n_target.detach())
        
        return loss

    def predict(self, xt, t, **kwargs):
        """
        Make prediction given noisy data, timestep t, and target timestep s.
        
        The model f(xt, t, s) learns to predict x_s.
        """
        # Target timestep `s` (e.g., t_n) is passed as the condition
        condition = kwargs.get('condition', None)
        if condition is None:
            # At inference, if no condition is given, assume target is x0 (t=0)
            condition = torch.zeros_like(t)
            
        if t.dtype!= xt.dtype:
            t = t.to(dtype=xt.dtype)
        if condition.dtype!= xt.dtype:
            condition = condition.to(dtype=xt.dtype)
            
        xt = xt.contiguous(memory_format=torch.channels_last)
        
        # The network MUST have use_additional_condition=True
        return self.network(xt, t, condition)
    
    def sample(self, shape, num_inference_timesteps=4, return_traj=False, verbose=False, **kwargs):
        """
        Generate samples from noise using K-step consistency sampling.
        This is what will be called by the official 'sampling.py' script.
        """
        device = self.device
        B = shape[0]
        # Start from x1 (pure noise)
        x = torch.randn(shape, device=device)
        x = x.contiguous(memory_format=torch.channels_last)
        traj = [x.clone()] if return_traj else None

        # Get the K-step uniform grid
        t_grid = torch.linspace(1.0, 0.0, steps=num_inference_timesteps + 1, device=device, dtype=x.dtype)
        
        for i in range(num_inference_timesteps):
            t_current = t_grid[i]
            t_target = t_grid[i+1]
            
            t_current_tensor = torch.full((B,), t_current, device=device, dtype=x.dtype)
            t_target_tensor = torch.full((B,), t_target, device=device, dtype=x.dtype)

            # Predict x_target directly
            # Model f(xt, t, s) predicts x_s
            x = self.predict(x, t_current_tensor, condition=t_target_tensor)
            x = x.contiguous(memory_format=torch.channels_last)

            if return_traj:
                traj.append(x.clone())

        x = x.clamp(-1.0, 1.0)
        return traj if return_traj else x


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_lcm_model(device="cpu", **kwargs):
    """
    Function to create the LCM (student) generative model.
    """
    
    # --- IMPORTANT ---
    # We MUST set use_additional_condition=True for this model
    # This is how the student model learns the consistency trajectory
    kwargs['use_additional_condition'] = True
    
    # Create U-Net backbone using the *original* function from custom_model.py
    # This ensures we use the exact same (unmodified) UNet architecture
    # We pass all kwargs (like compile) to the base creator
    base_model = create_base_network_model(device=device, **kwargs)
    network = base_model.network
    
    # Extract scheduler parameters
    scheduler_kwargs = dict(kwargs)
    num_train_timesteps = scheduler_kwargs.pop('num_train_timesteps', 1000)
    
    # Create the LCM scheduler
    scheduler = LCMScheduler(num_train_timesteps=num_train_timesteps, **scheduler_kwargs)
    
    # Create the LCM model wrapper
    model = CustomGenerativeModelLCM(network, scheduler, **kwargs)
    
    return model.to(device)