import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.LARTLoss import compute_alignment_loss, compute_temporal_loss, compute_reconstruction_loss

def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        align_labels: list = [1.0],
        mode: str = 'all',
        device: torch.device = 'cuda',
        
):
    """
    Train the model for one epoch.

    This function trains the model over all batches in the provided data loader. It calculates 
    various loss metrics (temporal, alignment, and reconstruction) for each batch and updates the model parameters.

    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): DataLoader providing the batches of data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        temporal_distances_mapping (dict): Mapping of timepoints to temporal distances.
        timepoints (list): List of timepoints to consider.
        align_labels (list): List of alignment labels for supervised loss computation.
        mode (str): Specifies which losses to combine ('align', 'temp', 'all').
        device (torch.device): Device to run the model on.

    Returns:
        dict: A dictionary containing average losses for the epoch (total, temporal, supervised, reconstruction).
    """
    model.train()

    # Lists to track different losses for the epoch
    loss_temp_epoch = []
    loss_sup_epoch = []
    loss_rec_epoch = []
    loss_total_epoch = []

    # Iterate over batches in the data loader
    for batch_data in tqdm(loader):
        # Iterate over all unique pairs of timepoints
        for i in range(len(timepoints)):
            for j in range(len(timepoints)):
                if i != j:
                    t1, t2, t3 = timepoints[i], timepoints[i], timepoints[j]

                    # Compute margin based on temporal distances
                    margin = np.abs(temporal_distances_mapping[t1] - temporal_distances_mapping[t3])

                    # Get the images from the batch for each timepoint
                    img_1 = batch_data[t1][0].float().to(device)
                    img_2 = batch_data[t2][1].float().to(device)
                    img_3 = batch_data[t3][0].float().to(device)
                    label = batch_data['pcr'].float().to(device)

                    # Forward pass through the model
                    rec1, latent_1 = model(img_1)
                    latent_1 = nn.functional.normalize(latent_1)
                    _, latent_2 = model(img_2)
                    latent_2 = nn.functional.normalize(latent_2)
                    rec3, latent_3 = model(img_3)
                    latent_3 = nn.functional.normalize(latent_3)

                    # Compute the temporal loss
                    loss_temp = compute_temporal_loss(latent_1, latent_2, latent_3, margin)

                    # Compute the alignment loss for the given labels
                    loss_align_total = compute_alignment_loss(latent_1, latent_2, label, align_labels)
                    loss_sup_epoch.append(loss_align_total.item())

                    # Compute reconstruction loss
                    loss_rec = compute_reconstruction_loss(rec1, rec3, batch_data[f'target_{t1}'], batch_data[f'target_{t3}'], device)

                    # Combine losses based on mode
                    if mode == 'align':
                        loss = loss_rec + loss_align_total
                    elif mode == 'temp':
                        loss = loss_rec + loss_temp
                    elif mode == 'all':
                        loss = loss_rec + loss_temp + loss_align_total

                    # Record the loss for each type
                    loss_temp_epoch.append(loss_temp.item())
                    loss_rec_epoch.append(loss_rec.item())

                    # Backpropagate and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Record total loss
                    loss_total_epoch.append(loss.item())

    # Return the average loss values for the epoch
    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'temp': sum(loss_temp_epoch) / len(loss_temp_epoch),
        'sup': sum(loss_sup_epoch) / len(loss_sup_epoch),
        'rec': sum(loss_rec_epoch) / len(loss_rec_epoch),
    }




def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        align_labels: list = [1.0],
        mode: str = 'all',
        device: torch.device = 'cuda',
        
):
    """
    Evaluate the model for one epoch.

    This function evaluates the model over all batches in the provided data loader. It calculates 
    various loss metrics (temporal, alignment, and reconstruction) for each batch without 
    updating the model parameters.

    Args:
        model (nn.Module): The model to be evaluated.
        loader (DataLoader): DataLoader providing the batches of data.
        temporal_distances_mapping (dict): Mapping of timepoints to temporal distances.
        timepoints (list): List of timepoints to consider.
        align_labels (list): List of alignment labels for supervised loss computation.
        mode (str): Specifies which losses to combine ('align', 'temp', 'all').
        device (torch.device): Device to run the model on.

    Returns:
        dict: A dictionary containing average losses for the epoch (temporal, supervised, reconstruction).
    """
    model.eval()

    # Lists to track different losses for the epoch
    loss_temp_epoch = []
    loss_sup_epoch = []
    loss_rec_epoch = []

    with torch.no_grad():
        # Iterate over batches in the data loader
        for batch_data in tqdm(loader):
            # Iterate over all unique pairs of timepoints
            for i in range(len(timepoints)):
                for j in range(len(timepoints)):
                    if i != j:
                        t1, t2, t3 = timepoints[i], timepoints[i], timepoints[j]

                        # Compute margin based on temporal distances
                        margin = np.abs(temporal_distances_mapping[t1] - temporal_distances_mapping[t3])

                        # Get the images from the batch for each timepoint
                        img_1 = batch_data[t1].float().to(device)
                        img_2 = batch_data[t2].float().to(device)
                        img_3 = batch_data[t3].float().to(device)
                        label = batch_data['pcr'].float().to(device)

                        # Forward pass through the model
                        rec1, latent_1 = model(img_1)
                        latent_1 = nn.functional.normalize(latent_1)
                        _, latent_2 = model(img_2)
                        latent_2 = nn.functional.normalize(latent_2)
                        rec3, latent_3 = model(img_3)
                        latent_3 = nn.functional.normalize(latent_3)

                        # Compute the temporal loss
                        loss_temp = compute_temporal_loss(latent_1, latent_2, latent_3, margin)

                        # Compute the alignment loss for the given labels
                        loss_align_total = compute_alignment_loss(latent_1, latent_2, label, align_labels)
                        loss_sup_epoch.append(loss_align_total.item())

                        # Compute reconstruction loss
                        loss_rec = compute_reconstruction_loss(rec1, rec3, batch_data[f'target_{t1}'], batch_data[f'target_{t3}'], device)

                        # Combine losses based on mode
                        if mode == 'align':
                            loss = loss_rec + loss_align_total
                        elif mode == 'temp':
                            loss = loss_rec + loss_temp
                        elif mode == 'all':
                            loss = loss_rec + loss_temp + loss_align_total

                        # Record the loss for each type
                        loss_temp_epoch.append(loss_temp.item())
                        loss_rec_epoch.append(loss_rec.item())

    # Return the average loss values for the epoch
    return {
        'temp': sum(loss_temp_epoch) / len(loss_temp_epoch),
        'sup': sum(loss_sup_epoch) / len(loss_sup_epoch),
        'rec': sum(loss_rec_epoch) / len(loss_rec_epoch),
    }
