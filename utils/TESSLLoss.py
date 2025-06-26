import torch
import torch.nn as nn
class TESSL_Loss(nn.Module):
    """
    TE-SSL Loss
    
    Based on Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, alpha=1, beta=0.9, compare_time_with_self=False):
        super(TESSL_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.alpha = alpha
        self.beta = beta
        self.compare_time_with_self = compare_time_with_self

    def forward(self, features, labels=None, mask=None, times=None):
        """
        Compute loss for model. 
        
        If `times` is None, Computes SupCon loss
        If `times`, `labels`, and `mask` are None, Computes SimCLR unsupervised loss:
            https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif times is not None and labels is None:
            raise ValueError('Cannot define "times" without defining "labels"')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        if times is not None:
            # Compute deltas
            a = times.tile((batch_size, 1))
            b = times.contiguous().view(-1, 1).tile(1, batch_size)
            deltas = torch.abs(a - b)
            # Get min and max of all positive pairs
            # inverted mask lets us alter only negative pairs
            inverted_mask = ((~torch.eq(labels, labels.T)).float())
            if self.compare_time_with_self:
                _min = torch.min(deltas + inverted_mask*1e10)
                _max = torch.min(deltas - inverted_mask*1e10)
            else:
                eye = torch.eye(batch_size).to(device)
                _mask = (inverted_mask + eye)*1e10
                _min = torch.min(deltas*mask + _mask)
                _max = torch.max(deltas*mask - _mask)

            m = (self.alpha - self.beta) / (_min - _max)
            b = (((self.beta - self.alpha) / (_min - _max))*_min) + self.alpha
            all_weights = (m * deltas) + b      # Weights for all pairs
            mask = all_weights * mask  # weights for only positive pairs, torch.equal(mask, (weighted_mask != 0).float()) should be true

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        log_prob_mask = logits_mask

        if times is not None:
            unmasked_weights = all_weights.repeat(anchor_count, contrast_count)
            log_prob_mask = unmasked_weights * logits_mask 

        # compute log_prob
        exp_logits = torch.exp(logits) * log_prob_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+ 1e-10)

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = (mask != 0).float().sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

