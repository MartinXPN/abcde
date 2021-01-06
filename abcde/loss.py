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
        """
        Compute pairwise ranking cross entropy loss:
        Cross entropy loss between the order of predicted pairs of vertices and
            the order of target pairs of vertices
            based on the betweenness
        :param pred_betweenness: predicted betweenness for each vertex in the graph
        :param target_betweenness: target betweenness fro each vertex in the graph
        :param src_ids: source ids in vertex pairs to compute ranking loss
        :param targ_ids: target ids in vertex pairs to compute ranking loss
        :return: CrossEntropy( predicted_pair_difference, target_pair_difference )
        """

        assert pred_betweenness.shape == target_betweenness.shape
        assert src_ids.shape == targ_ids.shape
        pred_diff = pred_betweenness[src_ids] - pred_betweenness[targ_ids]
        targ_diff = target_betweenness[src_ids] - target_betweenness[targ_ids]

        return F.binary_cross_entropy_with_logits(pred_diff, torch.sigmoid(targ_diff), reduction=self.reduction)
