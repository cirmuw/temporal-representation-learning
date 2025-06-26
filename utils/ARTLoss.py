import torch
import torch.nn as nn


def compute_temporal_loss(
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor, 
        margin: float
        ) -> torch.Tensor:
    """
    Compute the temporal loss on a patient level. 

    The triplet margin loss is computed as follows:
        L = max(d(a, p) - d(a, n) + margin, 0)

    where `d(x, y)` is the cosine similarity distance between `x` and `y`.

    Args:
        anchor (torch.Tensor): The anchor latent representation.
        positive (torch.Tensor): The positive latent representation.
        negative (torch.Tensor): The negative latent representation.
        margin (float): The margin value for the triplet loss.

    Returns:
        torch.Tensor: The computed temporal loss.
    """
    return nn.functional.triplet_margin_with_distance_loss(
        anchor=anchor,
        positive=positive,
        negative=negative,
        margin=margin,
        distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y),
        reduction='mean'
    )


def compute_alignment_loss(
        latents_1: torch.Tensor, 
        latents_2: torch.Tensor, 
        label: torch.Tensor, 
        align_labels: list
        ) -> torch.Tensor:
    """
    Compute alignment loss on a population level.

    The alignment loss is computed as the average of the cosine similarity between the
    latent representations of the same timepoint and same label.

    Args:
        latents_1 (torch.Tensor): Latent representations of the timepoint i.
        latents_2 (torch.Tensor): Latent representations of the timepoint i.
        label (torch.Tensor): The labels corresponding to the data.
        align_labels (list): Alignment labels to compute the loss for.

    Returns:
        torch.Tensor: The computed alignment loss averaged over the given labels.
    """
    loss_align_total = 0.0
    num_labels = len(align_labels)

    # Iterate over each alignment label and calculate the alignment loss
    for align_label in align_labels:
        # Select the latents for the current alignment label
        latents_1_label = latents_1[label == align_label]
        latents_2_label = latents_2[label == align_label]

        # Check if there are any valid instances for the current alignment label
        if latents_1_label.size(0) > 0 and latents_2_label.size(0) > 0:
            # Shuffle latents_1 to align on a population level with latents_2
            # This is done to ensure that the alignment loss is computed on the same instances
            # for both timepoints
            perm_latents_1 = torch.randperm(latents_1_label.size(0))
            latents_1_label = latents_1_label[perm_latents_1]

            # Compute the alignment loss as cosine similarity
            # The cosine similarity is a measure of the similarity between two vectors
            # It is defined as the dot product of the two vectors divided by the product
            # of their magnitudes
            loss_align = 1.0 - nn.functional.cosine_similarity(latents_1_label, latents_2_label, dim=-1).mean()
            loss_align_total += loss_align

    # Compute the average alignment loss across all labels
    if num_labels > 0:
        loss_align_total /= num_labels

    return loss_align_total



def compute_reconstruction_loss(rec1, rec3, target_1, target_3, device) -> torch.Tensor:
    """
    Compute the reconstruction loss.
    """
    rec_img = torch.cat([rec1, rec3], dim=0)
    target_img = torch.cat([target_1, target_3], dim=0).to(device)
    return nn.functional.mse_loss(rec_img, target_img, reduction='mean')
