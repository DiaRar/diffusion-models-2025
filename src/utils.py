import random
import torch
import json
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from datetime import datetime

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now
    
def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Expand timestep tensor to match the spatial dimensions of x.
    
    Args:
        t: Timestep tensor of shape (batch_size,)
        x: Data tensor of shape (batch_size, channels, height, width)
        
    Returns:
        Expanded timestep tensor
    """
    for _ in range(x.ndim - 1):
        t = t.unsqueeze(-1)
    return t

def tensor_to_pil_image(tensor: torch.Tensor):
    """
    Convert tensor to PIL image.
    
    Args:
        tensor: Tensor of shape (batch_size, channels, height, width) or (channels, height, width)
    Returns:
        List of PIL images or single PIL image if a batch size is 1.
    """
    # Handle 3D tensors (single image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    single_image = tensor.shape[0] == 1

    # denormalize from [-1, 1] to [0, 1]
    tensor = (tensor * 0.5 + 0.5)
    
    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL
    tensor = tensor.clone().detach().cpu().permute(0,2,3,1).numpy()
    images = (tensor * 255).round().astype("uint8")
    images = [Image.fromarray(image) for image in images]
    if single_image:
        return images[0]
    return images


def save_model(model, checkpoint_path: str, model_config: Dict[str, Any], config_path: str = None):
    """
    Save model checkpoint and configuration JSON file together.
    
    Args:
        model: The generative model to save
        checkpoint_path: Path to save the checkpoint (.pt file)
        model_config: Dictionary containing model configuration
        config_path: Path to save the config JSON file (default: same directory as checkpoint)
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    use_xla = False
    try:
        import torch_xla.core.xla_model as xm
        use_xla = any(p.device.type == 'xla' for p in model.parameters())
    except Exception:
        use_xla = False
    if use_xla:
        import torch_xla.core.xla_model as xm
        xm.save({"state_dict": model.state_dict()}, str(checkpoint_path))
    else:
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    
    # Save model config JSON
    if config_path is None:
        config_path = checkpoint_path.parent / 'model_config.json'
    else:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)


def load_model(checkpoint_path: str, create_model_fn, device: str = "cpu", config_path: str = None):
    """
    Load model from checkpoint and configuration JSON file.
    
    Args:
        checkpoint_path: Path to the checkpoint (.pt file)
        create_model_fn: Function to create the model (e.g., create_custom_model)
        device: Device to load the model on
        config_path: Path to the config JSON file (default: same directory as checkpoint)
        
    Returns:
        Loaded model
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Determine config path
    if config_path is None:
        config_path = checkpoint_path.parent / 'model_config.json'
    else:
        config_path = Path(config_path)
    
    # Load model config
    model_kwargs = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        for key, value in saved_config.items():
            model_kwargs[key] = value
    
    # Create model architecture
    model = create_model_fn(device=device, **model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    return model