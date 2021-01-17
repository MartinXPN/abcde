import os
from dataclasses import dataclass, field, InitVar
from typing import List, Any, Union, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from igraph import Graph
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
from tqdm import trange


@dataclass
class RandomGraphs(Dataset[Data]):
    min_nodes: int
    max_nodes: int
    graph_type: str = 'powerlaw'
    nb_graphs: int = 1
    repeats: int = 1
    graphs: List[Data] = field(default_factory=list)
    verbose: InitVar[bool] = True

    def __post_init__(self, verbose):
        assert self.repeats >= 1
        for _ in trange(self.nb_graphs, disable=not verbose):
            # Generate a random graph with NetworkX
            cur_n = np.random.randint(self.min_nodes, self.max_nodes)
            if self.graph_type == 'erdos_renyi':        graph = nx.erdos_renyi_graph(n=cur_n, p=0.15)
            elif self.graph_type == 'small-world':      graph = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
            elif self.graph_type == 'barabasi_albert':  graph = nx.barabasi_albert_graph(n=cur_n, m=4)
            elif self.graph_type == 'powerlaw':         graph = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
            else:
                raise ValueError(f'{self.graph_type} graph type is not supported yet')

            # Convert the NetworkX graph to iGraph
            g = Graph(directed=False)
            g.add_vertices(graph.nodes())
            g.add_edges(graph.edges())

            betweenness = np.expand_dims(g.betweenness(directed=False), -1)

            degrees = nx.degree_centrality(graph)
            degrees = np.array([degrees[n] for n in range(cur_n)], dtype='float32')
            degrees = np.expand_dims(degrees, -1)

            src_ids, tgt_ids = self.gen_vertex_pairs(n=cur_n, size=5 * cur_n)
            edge_index = np.array(graph.to_directed(as_view=True).edges).T
            res = Data(x=torch.from_numpy(degrees),
                       y=torch.from_numpy(betweenness),
                       src_ids=torch.from_numpy(src_ids),
                       tgt_ids=torch.from_numpy(tgt_ids),
                       edge_index=torch.from_numpy(edge_index))
            self.graphs.append(res)

    @staticmethod
    def gen_vertex_pairs(n: int, size: int) -> Tuple[np.array, np.array]:
        src_ids = np.hstack([np.arange(n), np.random.randint(low=0, high=n, size=size - n)])
        tgt_ids = np.hstack([np.arange(1, n + 1) % n, np.random.randint(low=0, high=n, size=size - n)])
        return src_ids, tgt_ids

    def __getitem__(self, index) -> Data:
        index %= len(self.graphs)
        return self.graphs[index]

    def __len__(self) -> int:
        return self.nb_graphs * self.repeats


class GraphDataModule(LightningDataModule):
    train_dataset: RandomGraphs
    valid_dataset: RandomGraphs

    def __init__(self,
                 min_nodes: int, max_nodes: int, nb_train_graphs: int, nb_valid_graphs: int,
                 batch_size: int, graph_type: str = 'powerlaw', repeats: int = 8, regenerate_epoch_interval: int = 5,
                 verbose: bool = True,
                 train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms, val_transforms, test_transforms, dims=dims)
        self.min_nodes: int = min_nodes
        self.max_nodes: int = max_nodes
        self.nb_train_graphs: int = nb_train_graphs
        self.nb_valid_graphs: int = nb_valid_graphs
        self.batch_size: int = batch_size
        self.graph_type: str = graph_type
        self.repeats: int = repeats
        self.regenerate_every_epochs: int = regenerate_epoch_interval
        self.verbose: bool = verbose
        self.train_epochs: int = 0
        self.valid_epochs: int = 0
        self.workers: int = max(os.cpu_count() - 1, 1)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """ Generate random graphs and return the loader for those """
        if self.train_epochs % self.regenerate_every_epochs == 0:
            print(f'Generating {self.nb_train_graphs} new Train graphs - [{self.min_nodes}-{self.max_nodes}]...')
            self.train_dataset = RandomGraphs(min_nodes=self.min_nodes, max_nodes=self.max_nodes,
                                              nb_graphs=self.nb_train_graphs, repeats=self.repeats,
                                              verbose=self.verbose)
        self.train_epochs += 1
        return DataLoader(self.train_dataset.graphs, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """ Generate random graphs and return the loader for those """
        if self.valid_epochs % self.regenerate_every_epochs == 0:
            print(f'Generating {self.nb_valid_graphs} new Validation graphs - [{self.min_nodes}-{self.max_nodes}]...')
            self.valid_dataset = RandomGraphs(min_nodes=self.min_nodes, max_nodes=self.max_nodes,
                                              nb_graphs=self.nb_valid_graphs, repeats=1, verbose=self.verbose)
        self.valid_epochs += 1
        return DataLoader(self.valid_dataset.graphs, batch_size=1, shuffle=False, num_workers=self.workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        batch.to(device)
        return batch
