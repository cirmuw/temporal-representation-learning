import torch
import numpy as np
import random
 
def seed_worker(worker_id):
    """
    Ensures that each DataLoader worker has a unique and reproducible seed.
 
    Args:
        worker_id (int): ID of the worker in DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_all(seed_value):
    """
    Sets seeds across PyTorch, NumPy, and Python's `random` to ensure reproducibility.
     
    Args:
        seed_value (int): The seed value to use for all libraries.
    """
    # 1. Set the random seed for PyTorch
    torch.manual_seed(seed_value)
 
    # 2. Set the seed for NumPy and Python's built-in random library
    np.random.seed(seed_value)
    random.seed(seed_value)
 
    # 3. If using CUDA, set seeds for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU

    # 4. If using the MPS backend, ensure deterministic behavior
    if torch.backends.mps.is_available():
        torch.manual_seed(seed_value)  # MPS uses PyTorch's manual seed
 
 
    # 4. Ensure deterministic behavior for GPU operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
