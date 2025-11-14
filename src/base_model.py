"""
Abstract base classes for diffusion and flow models
Students need to implement concrete classes that inherit from these base classes.

This module provides a general interface inspired by EDM and other well-designed
diffusion model repositories, supporting both diffusion and flow-based models.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import numpy as np


class BaseScheduler(ABC, nn.Module):
    """
    Abstract base class for noise schedulers.
    Provides a general interface for diffusion/flow models.
    """
    
    def __init__(self, num_train_timesteps: int = 1000, **kwargs):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self._setup(**kwargs)
    
    def _setup(self, **kwargs):
        """
        Setup scheduler-specific parameters. Override in subclasses.
        e.g., betas, alphas, alphas_cumprod, etc.
        """
        
        pass
    
    @abstractmethod
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample random timesteps for training.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to place timesteps on
            
        Returns:
            Tensor of shape (batch_size,) with timestep values
        """
        pass
    
    @abstractmethod
    def forward_process(self, data: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward process to transform clean data to noisy data.
        
        Args:
            data: Clean data tensor (batch_size, channels, height, width)
            noise: Noise tensor (batch_size, channels, height, width) 
            t: Timestep tensor (batch_size,)
            
        Returns:
            Noisy data tensor (batch_size, channels, height, width)
        """
        pass
    
    @abstractmethod
    def reverse_process_step(self, xt: torch.Tensor, pred: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        """
        Perform one step of the reverse (denoising) process.
        
        Args:
            xt: Current noisy data (batch_size, channels, height, width)
            pred: Model prediction (batch_size, channels, height, width)
            t: Current timestep (batch_size,)
            t_next: Next timestep (batch_size,)
            
        Returns:
            Updated data tensor at timestep t_next (batch_size, channels, height, width)
        """
        pass
    
    @abstractmethod
    def get_target(self, data: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get the target for model prediction used in L2 loss computation.
        
        e.g., for diffusion models: target = noise (for noise prediction) or x0 (for x0 prediction)
              for flow models: target = velocity field
        
        Args:
            data: Clean data tensor
            noise: Noise tensor  
            t: Timestep tensor
            
        Returns:
            Target tensor for model prediction
        """
        pass
    

class BaseGenerativeModel(ABC, nn.Module):
    """
    Abstract base class for generative models (diffusion, flow, etc.).
    Provides a unified interface for different types of generative models.
    """
    
    def __init__(self, network: nn.Module, scheduler: BaseScheduler, **kwargs):
        super().__init__()
        self.network = network
        self.scheduler = scheduler
        self._setup(**kwargs)
    
    def _setup(self, **kwargs):
        """Setup model-specific parameters. Override in subclasses."""
        pass
    
    @property
    def device(self):
        """Get the device of the model parameters."""
        return next(self.network.parameters()).device
    
    @abstractmethod
    def compute_loss(self, data: torch.Tensor, noise: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the training loss.
        
        Args:
            data: Data samples (clean images)
            noise: Prior samples (noise for diffusion, x0 for flow)
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def predict(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Make a prediction given noisy data and timestep.
        
        Args:
            xt: Noisy data tensor
            t: Timestep tensor
            **kwargs: Additional arguments (e.g., aux_timestep for some advanced models)
            
        Returns:
            Model prediction tensor
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_timesteps: int = 20,
        return_traj: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate samples from the model.
        
        Args:
            shape: Shape of the samples to generate (batch_size, channels, height, width)
            num_inference_timesteps: Number of inference steps
            return_traj: Whether to return the full trajectory
            verbose: Whether to show progress
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Generated samples or trajectory
        """
        pass