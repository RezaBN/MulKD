# losses.py
"""
This file contains the custom loss components for Contrastive Representation
Distillation (CRD), including the projection head and the NCE-based loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE

class CRDProjectionHead(nn.Module):
    """
    A projection head used in CRD to map features to a lower-dimensional space
    where the contrastive loss is calculated.
    """
    def __init__(self, input_dim, output_dim=128):
        super(CRDProjectionHead, self).__init__()
        # Define a hidden dimension, which is a common practice in projection heads
        hidden_dim = output_dim
        if input_dim > 0:
            hidden_dim = max(output_dim, input_dim // 2)
        else:
            print(f"Warning: CRDProjectionHead received input_dim <= 0 ({input_dim}). Defaulting.")
            input_dim = output_dim
            hidden_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # L2-normalize the output features
        return F.normalize(x, p=2, dim=1)


class NCELoss(nn.Module):
    """
    Noise-Contrastive Estimation Loss. This implementation uses in-batch negatives.
    It encourages the features of a student-teacher pair (positive pair) to be more
    similar than features of a student and other teachers in the batch (negative pairs).
    """
    def __init__(self, temperature, num_negative_samples_config):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        # The number of negative samples is implicitly defined by the batch size in this implementation.
        # self.num_negative_samples = num_negative_samples_config

    def forward(self, feat_s, feat_t):
        batch_size = feat_s.shape[0]

        # Positive pairs: cosine similarity between corresponding student and teacher features
        l_pos = torch.einsum('nc,nc->n', [feat_s, feat_t]).unsqueeze(-1)  # Shape: [B, 1]

        # Negative pairs: cosine similarity of each student feature with all teacher features in the batch
        l_neg_all_pairs = torch.einsum('nc,kc->nk', [feat_s, feat_t])  # Shape: [B, B]

        # Mask to exclude the positive pairs (diagonal) from the negative pairs
        mask = torch.eye(batch_size, dtype=torch.bool).to(DEVICE)
        l_neg = l_neg_all_pairs.masked_fill(mask, -float('inf'))

        # Concatenate positive and negative logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # The target label is always 0, as the positive logit is the first one
        labels = torch.zeros(batch_size, dtype=torch.long).to(DEVICE)

        return F.cross_entropy(logits, labels)


class CRDLoss(nn.Module):
    """
    Contrastive Representation Distillation (CRD) loss module.
    This module contains projection heads for both student and teacher features
    and computes the NCE loss between the projected features.
    """
    def __init__(self, opt_crd_params):
        super(CRDLoss, self).__init__()
        self.embed_s = CRDProjectionHead(input_dim=opt_crd_params.s_dim, output_dim=opt_crd_params.feat_dim)
        self.embed_t = CRDProjectionHead(input_dim=opt_crd_params.t_dim, output_dim=opt_crd_params.feat_dim)
        self.criterion = NCELoss(opt_crd_params.nce_t, opt_crd_params.nce_n)

    def forward(self, f_s, f_t):
        """
        Calculates the CRD loss.
        f_s: student features
        f_t: teacher features
        """
        # Project features into the contrastive space
        f_s_projected = self.embed_s(f_s)
        f_t_projected = self.embed_t(f_t)

        # Calculate NCE loss. Teacher features are detached to prevent gradients from flowing back to the teacher.
        loss = self.criterion(f_s_projected, f_t_projected.detach())
        return loss
