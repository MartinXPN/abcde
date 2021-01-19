import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_squared_error, max_error
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj

from abcde.loss import PairwiseRankingCrossEntropyLoss
from abcde.metrics import kendall_tau, top_k_ranking_accuracy


class BetweennessCentralityEstimator(pl.LightningModule):

    def __init__(self, lr_reduce_patience: int = 1):
        super().__init__()
        self.lr_reduce_patience: int = lr_reduce_patience
        self.criterion = PairwiseRankingCrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs = batch
        pred = self(inputs)

        loss = self.criterion(pred, inputs.y, inputs.tgt_ids, inputs.src_ids)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Has to process the graphs one by one as the predictions and labels are sorted at once """
        with torch.no_grad():
            pred: np.ndarray = self(batch).cpu().detach().numpy()
            label: np.ndarray = batch.y.cpu().detach().numpy()
            degrees: np.ndarray = batch.x.cpu().detach().numpy()

        # Compute metrics for each graph in the batch
        graphs = batch.to_data_list() if isinstance(batch, Batch) else [batch]
        start = 0
        for g in graphs:
            end = start + g.num_nodes
            p = pred[start: end].flatten()
            l = label[start: end].flatten()
            d = degrees[start: end].flatten()
            start = end

            # Vertices with deg(v) <= 1 (leafs) can't have high betweenness-centrality
            mask = d * len(d) < 1.1
            p[mask] = p.min() - np.finfo(np.float32).eps

            top_pred = np.argsort(-p)
            top_label = np.argsort(-l)
            res = {
                'val_top_1%': top_k_ranking_accuracy(top_label, top_pred, k=0.01) * 100,
                'val_top_5%': top_k_ranking_accuracy(top_label, top_pred, k=0.05) * 100,
                'val_top_10%': top_k_ranking_accuracy(top_label, top_pred, k=0.1) * 100,
                'val_kendal': kendall_tau(l, p) * 100,
                'val_mse': mean_squared_error(l, p),
                'val_max_error': max_error(l, p)
            }
            self.log_dict(res)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=self.lr_reduce_patience, factor=0.7)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_kendal'
        }


class DrBC(BetweennessCentralityEstimator):
    def __init__(self, nb_gcn_cycles: int = 5, lr_reduce_patience: int = 1):
        super().__init__(lr_reduce_patience=lr_reduce_patience)
        self.nb_gcn_cycles: int = nb_gcn_cycles

        self.node_linear = nn.Linear(1, 128)
        self.conv = GCNConv(128, 128)
        self.gru = nn.GRUCell(128, 128)
        self.linear2 = nn.Linear(129, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, inputs):
        node_features, edge_index = inputs.x, inputs.edge_index
        node_features = self.node_linear(node_features)
        node_features = F.leaky_relu(node_features, negative_slope=0.3)
        node_features = F.normalize(node_features, p=2, dim=-1)

        states = [node_features]
        for rep in range(self.nb_gcn_cycles):
            x = self.conv(states[-1], edge_index)
            x = self.gru(x, states[-1])
            x = F.normalize(x, p=2, dim=-1)
            states.append(x)

        x = states[-1]
        # x = torch.cat([x, node_features], dim=-1)
        x = torch.cat([x, inputs.x], dim=-1)
        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.linear3(x)
        return x


class ABCDE(BetweennessCentralityEstimator):
    def __init__(self, nb_gcn_cycles: int = 5, lr_reduce_patience: int = 1):
        super().__init__(lr_reduce_patience=lr_reduce_patience)
        self.nb_gcn_cycles: int = nb_gcn_cycles

        self.node_linear = nn.Linear(1, 32)
        self.linear1 = nn.Linear(32, 128)
        # self.convolutions = nn.ModuleList([GCNConv(128, 128, improved=True) for _ in range(nb_gcn_cycles)])
        # self.conv = GATConv(128, out_channels=128 // 4, heads=4, negative_slope=0.3)
        self.conv = GCNConv(128, 128)
        self.gru = nn.GRUCell(128, 128)
        self.linear2 = nn.Linear(128 + 32 + 1, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, inputs):
        node_features, edge_index = inputs.x, inputs.edge_index
        drop_edge, _ = dropout_adj(edge_index, p=0.3, force_undirected=True, training=self.training)
        node_features = self.node_linear(node_features)
        node_features = F.leaky_relu(node_features, negative_slope=0.3)
        node_features = F.normalize(node_features, p=2, dim=-1)

        x = self.linear1(node_features)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = F.normalize(x, p=2, dim=-1)
        states = [x]
        # for conv in self.convolutions:
        for rep in range(self.nb_gcn_cycles):
            x = self.conv(x, drop_edge)
            x = self.gru(x, states[-1])
            x = F.normalize(x, p=2, dim=-1)
            states.append(x)

        x = torch.cat([states[-1], node_features, inputs.x], dim=-1)
        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.linear3(x)
        return x
