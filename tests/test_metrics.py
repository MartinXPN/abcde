from unittest import TestCase

import numpy as np
from sklearn.metrics import mean_squared_error, max_error

from abcde.metrics import top_k_ranking_accuracy, kendall_tau


class TestMetrics(TestCase):

    def test_empty(self):
        self.assertTrue(np.isnan(kendall_tau([], [])))

    def test_wrong_dims(self):
        with self.assertRaises(ValueError):
            kendall_tau([1, 2], [1])

    def test_simple(self):
        label = np.array([1, 2, 3])
        pred = np.array([2, 3, 4])
        top_pred = np.argsort(-pred)
        top_label = np.argsort(-label)

        self.assertEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.01), 1)
        self.assertEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.05), 1)
        self.assertEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.1), 1)
        self.assertEqual(kendall_tau(label, pred), 1)
        self.assertEqual(mean_squared_error(label, pred), 1)
        self.assertEqual(max_error(label, pred), 1)

    def test_perfect_top_k(self):
        label = np.arange(1, 100)
        pred = np.arange(1, 100)
        pred[3] = 0

        top_pred = np.argsort(-pred)
        top_label = np.argsort(-label)

        self.assertAlmostEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.01), 1)
        self.assertAlmostEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.05), 1)
        self.assertAlmostEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.1), 1)
        self.assertAlmostEqual(kendall_tau(label, pred), 0.9987631416202845)
        self.assertAlmostEqual(mean_squared_error(label, pred), 0.16161616161616163)
        self.assertAlmostEqual(max_error(label, pred), 4)

    def test_mixed_top_k(self):
        label = np.arange(1, 100)
        pred = np.arange(1, 100)
        pred[98] = 0

        top_pred = np.argsort(-pred)
        top_label = np.argsort(-label)

        self.assertAlmostEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.01), 0)
        self.assertAlmostEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.05), 0.75)
        self.assertAlmostEqual(top_k_ranking_accuracy(top_label, top_pred, k=0.1), 0.8888888888888888)
        self.assertAlmostEqual(kendall_tau(label, pred), 0.9595959595959596)
        self.assertAlmostEqual(mean_squared_error(label, pred), 99)
        self.assertAlmostEqual(max_error(label, pred), 99)
