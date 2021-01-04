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
        pred_betweenness = pred_betweenness.squeeze(-1)
        target_betweenness = target_betweenness.squeeze(-1)

        assert pred_betweenness.shape == target_betweenness.shape
        assert src_ids.shape == targ_ids.shape
        pred_diff = pred_betweenness[src_ids] - pred_betweenness[targ_ids]
        targ_diff = target_betweenness[src_ids] - target_betweenness[targ_ids]

        pred_diff = torch.sigmoid(pred_diff).unsqueeze(-1)
        targ_diff = torch.sigmoid(targ_diff).unsqueeze(-1)

        return F.binary_cross_entropy_with_logits(pred_diff, targ_diff, reduction=self.reduction)


# loss = PairwiseRankingCrossEntropyLoss()
# print(loss(pred_betweenness=Tensor([0.5, 0.7, 3]), target_betweenness=Tensor([0.2, 1, 2]),
#            src_ids=Tensor([0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 2, 1, ]).long(),
#            targ_ids=Tensor([1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, ]).long()))
