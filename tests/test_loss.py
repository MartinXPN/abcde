from unittest import TestCase

from torch import Tensor

from abcde.loss import PairwiseRankingCrossEntropyLoss


class TestPairwiseRankingLoss(TestCase):
    def test_simple_case(self):
        loss = PairwiseRankingCrossEntropyLoss()
        res = loss(pred_betweenness=Tensor([[0.5], [0.7], [3]]), target_betweenness=Tensor([[0.2], [1], [2]]),
                   src_ids=Tensor([0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 2, 1, ]).long(),
                   targ_ids=Tensor([1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, ]).long())
        # This number is taken from the tensorflow implementation
        self.assertAlmostEqual(res, 0.636405362070762)
