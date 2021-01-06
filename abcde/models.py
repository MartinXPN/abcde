import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_squared_error, max_error
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv

from abcde.loss import PairwiseRankingCrossEntropyLoss
from abcde.util import kendall_tau, top_k_ranking_accuracy


class ABCDE(pl.LightningModule):
    def __init__(self, nb_gcn_cycles: int = 5, eval_interval: int = 1):
        super().__init__()
        self.eval_interval = eval_interval
        self.node_linear = nn.Linear(1, 128)
        self.convolutions = nn.ModuleList([GCNConv(128, 128) for _ in range(nb_gcn_cycles)])
        self.gru = nn.GRUCell(128, 128)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 1)
        self.criterion = PairwiseRankingCrossEntropyLoss()

    def forward(self, inputs):
        node_features, edge_index = inputs.x, inputs.edge_index
        node_features = self.node_linear(node_features)
        node_features = F.leaky_relu(node_features)
        node_features = F.normalize(node_features, p=2, dim=1)

        x = node_features
        for conv in self.convolutions:
            x = conv(x, edge_index)
            x = self.gru(x)
            x = F.normalize(x, p=2, dim=1)

        x = torch.cat([x, node_features], dim=-1)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.linear3(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch
        pred = self(inputs)

        loss = self.criterion(pred, inputs.y, inputs.tgt_ids, inputs.src_ids)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Has to process the graphs one by one as the predictions and labels are sorted at once """
        with torch.no_grad():
            pred = self(batch).cpu().detach().numpy().flatten()
            label = batch.y.cpu().detach().numpy().flatten()

        top_pred = np.argsort(-pred)
        top_label = np.argsort(-label)

        self.log('val_top_0.01%', top_k_ranking_accuracy(top_label, top_pred, k=0.01))
        self.log('val_top_0.5%', top_k_ranking_accuracy(top_label, top_pred, k=0.05))
        self.log('val_top_1%', top_k_ranking_accuracy(top_label, top_pred, k=0.1))
        self.log('val_kendal', kendall_tau(label, pred))
        self.log('val_mse', mean_squared_error(label, pred))
        self.log('val_max_error', max_error(label, pred))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3 * self.eval_interval, factor=0.7)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_kendal'
        }
