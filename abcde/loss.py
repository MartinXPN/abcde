import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class PairwiseRankingCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction: str = reduction

    def forward(self, pred_betweenness: Tensor, target_betweenness: Tensor,
                src_ids: Tensor, targ_ids: Tensor) -> Tensor:

        assert pred_betweenness.shape == target_betweenness.shape
        assert src_ids.shape == targ_ids.shape
        pred_diff = pred_betweenness[src_ids] - pred_betweenness[targ_ids]
        targ_diff = target_betweenness[src_ids] - target_betweenness[targ_ids]

        return F.binary_cross_entropy_with_logits(pred_diff, torch.sigmoid(targ_diff), reduction=self.reduction)
