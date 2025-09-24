import random
import numpy as np
import torch


def set_random_seed(seed=42):
    """
    Sets the random seed for reproducibility
    """
    print(f"🎲 Setting random seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # For full determinism (can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("✅ Random seed set for all libraries")