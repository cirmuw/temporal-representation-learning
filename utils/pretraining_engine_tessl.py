import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm


def compute_tessl_loss(features, events, times, criterion_tessl):
    """
    Compute the Tessl loss.
    """
    return criterion_tessl(features, labels=events, times=times)


def compute_reconstruction_loss(rec_img, targets):
    """
    Compute the reconstruction loss (MSE).
    """
    return nn.functional.mse_loss(rec_img, targets, reduction='mean')


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion_tessl: nn.Module,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        device: torch.device = 'cuda',
        mode: str = 'all',
):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): DataLoader providing the batches of data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion_tessl (nn.Module): Criterion for calculating Tessl loss.
        temporal_distances_mapping (dict): Mapping of timepoints to temporal distances.
        timepoints (list): List of timepoints to consider.
        device (torch.device): Device to run the model on.
        mode (str): Specifies which losses to combine ('align', 'temp', 'all').

    Returns:
        dict: A dictionary containing average losses for the epoch (total, tessl, reconstruction).
    """
    model.train()

    # Lists to track different losses for the epoch
    loss_tessl_epoch = []
    loss_rec_epoch = []
    loss_total_epoch = []
    
    acc_steps = 8
    for i, batch_data in enumerate(tqdm(loader)):
        bsz = batch_data['T0'][0].shape[0]
        data_view1 = torch.cat([batch_data[t][0] for t in timepoints], dim=0).to(device)
        data_view2 = torch.cat([batch_data[t][1] for t in timepoints], dim=0).to(device)
        targets = torch.cat([batch_data[f'target_{t}'] for t in timepoints], dim=0).to(device)
        events = torch.cat([batch_data['pcr'] for _ in range(4)], dim=0).to(device)
        times = torch.tensor([temporal_distances_mapping[t] for t in timepoints for _ in range(bsz)]).to(device)

        # Forward pass
        rec1, out_1 = model(data_view1.float())
        _, out_2 = model(data_view2.float())
        features = torch.cat([nn.functional.normalize(out_1.unsqueeze(1)), nn.functional.normalize(out_2.unsqueeze(1))], 1)

        # Compute Tessl loss
        loss_tessl = compute_tessl_loss(features, events, times, criterion_tessl)
        
        # Compute reconstruction loss
        loss_rec = compute_reconstruction_loss(rec1, targets)

        # Combine losses based on mode
        if mode == 'all':
            loss = loss_tessl + loss_rec
        elif mode == 'no_rec':
            loss = loss_tessl

        loss_tessl_epoch.append(loss_tessl.item())
        loss_rec_epoch.append(loss_rec.item())
        loss_total_epoch.append(loss.item())

        # Normalize gradients over accumulation steps
        loss = loss / acc_steps
        loss.backward()

        if i % acc_steps == 0 or i == len(loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

    return {
        'total': sum(loss_total_epoch) / len(loss_total_epoch),
        'rec': sum(loss_rec_epoch) / len(loss_rec_epoch),
        'tessl': sum(loss_tessl_epoch) / len(loss_tessl_epoch)
    }

def eval_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion_tessl: nn.Module,
        temporal_distances_mapping: dict = {'T0': 1.0, 'T1': 0.75, 'T2': 0.5, 'T3': 0.25},
        timepoints: list = ['T0', 'T1', 'T2', 'T3'],
        device: torch.device = 'cuda',
):
    """
    Evaluate the model for one epoch.

    Args:
        model (nn.Module): The model to be evaluated.
        loader (DataLoader): DataLoader providing the batches of data.
        criterion_tessl (nn.Module): Criterion for calculating Tessl loss.
        temporal_distances_mapping (dict): Mapping of timepoints to temporal distances.
        timepoints (list): List of timepoints to consider.
        device (torch.device): Device to run the model on.

    Returns:
        dict: A dictionary containing average losses for the epoch (tessl, reconstruction).
    """
    model.eval()

    # Lists to track different losses for the epoch
    loss_tessl_epoch = []
    loss_rec_epoch = []

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(loader)):
            bsz = batch_data['T0'][0].shape[0]
            data_view1 = torch.cat([batch_data[t][0] for t in timepoints], dim=0).to(device)
            data_view2 = torch.cat([batch_data[t][1] for t in timepoints], dim=0).to(device)
            targets = torch.cat([batch_data[f'target_{t}'] for t in timepoints], dim=0).to(device)
            events = torch.cat([batch_data['pcr'] for _ in range(4)], dim=0).to(device)
            times = torch.tensor([temporal_distances_mapping[t] for t in timepoints for _ in range(bsz)]).to(device)

            # Forward pass
            rec1, out_1 = model(data_view1.float())
            _, out_2 = model(data_view2.float())
            features = torch.cat([nn.functional.normalize(out_1.unsqueeze(1)), nn.functional.normalize(out_2.unsqueeze(1))], 1)

            # Compute Tessl loss
            loss_tessl = compute_tessl_loss(features, events, times, criterion_tessl)

            # Compute reconstruction loss
            loss_rec = compute_reconstruction_loss(rec1, targets)

            loss_tessl_epoch.append(loss_tessl.item())
            loss_rec_epoch.append(loss_rec.item())

    return {
        'rec': sum(loss_rec_epoch) / len(loss_rec_epoch),
        'tessl': sum(loss_tessl_epoch) / len(loss_tessl_epoch)
    }
