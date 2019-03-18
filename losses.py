import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss



class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, inputs, labels, logits, recons):
        batch_size = inputs.shape[0]

        left = F.relu(0.9 - logits, inplace=True) ** 2
        right = F.relu(logits - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        recons_loss = ((inputs.view(batch_size, -1) - recons)**2).sum()

        return (margin_loss + 0.0005 * recons_loss) / batch_size
