import os
import time
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import List, Any, Union, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from igraph import Graph
from pytorch_lightning import LightningDataModule
from torch.multiprocessing import Pool
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader

from abcde.util import ThreadWithReturnValue


@dataclass
class RandomGraphs(Dataset[Data]):
    min_nodes: int
    max_nodes: int
    graph_type: str = 'powerlaw'
    nb_graphs: int = 1
    repeats: int = 1
    graphs: List[Data] = field(default_factory=list)
    save_path: InitVar[Union[Path, None]] = None

    def __post_init__(self, save_path):
        assert self.repeats >= 1
        workers: int = max(os.cpu_count(), 1)

        if save_path is not None and save_path.exists():
            print('Found existing dataset at', save_path, 'loading it...')
            self.graphs = torch.load(save_path)
            return

        start = time.time()
        with Pool(workers) as p:
            nb_nodes = np.random.randint(self.min_nodes, self.max_nodes, size=self.nb_graphs)
            # 'small-world', 'barabasi_albert', p=[0.3, 0.3, 0.4]
            graph_types = np.random.choice([self.graph_type], size=self.nb_graphs, replace=True)
            self.graphs = p.starmap(self.generate_graph, zip(nb_nodes, graph_types))

        print(f'Generated {len(self.graphs)} graphs in {time.time() - start}s')
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.graphs, save_path)

    def __getitem__(self, index) -> Data:
        index %= len(self.graphs)
        return self.graphs[index]

    def __len__(self) -> int:
        return self.nb_graphs * self.repeats

    @staticmethod
    def gen_vertex_pairs(n: int, size: int) -> Tuple[np.array, np.array]:
        src_ids = np.hstack([np.arange(n), np.random.randint(low=0, high=n, size=size - n)])
        tgt_ids = np.hstack([np.arange(1, n + 1) % n, np.random.randint(low=0, high=n, size=size - n)])
        return src_ids, tgt_ids

    @staticmethod
    def generate_graph(n: int, graph_type: str) -> Data:
        # Generate a random graph with NetworkX
        if graph_type == 'erdos_renyi':         graph = nx.erdos_renyi_graph(n, p=4 / n)
        elif graph_type == 'small-world':       graph = nx.connected_watts_strogatz_graph(n, k=4, p=0.1)
        elif graph_type == 'barabasi_albert':   graph = nx.barabasi_albert_graph(n, m=4)
        elif graph_type == 'powerlaw':          graph = nx.powerlaw_cluster_graph(n, m=4, p=0.05)
        else:
            raise ValueError(f'{graph_type} graph type is not supported yet')

        # Convert the NetworkX graph to iGraph
        g = Graph(directed=False)
        g.add_vertices(graph.nodes())
        g.add_edges(graph.edges())

        betweenness = np.expand_dims(g.betweenness(directed=False), -1)

        degrees = nx.degree_centrality(graph)
        degrees = np.array([degrees[n] for n in range(n)], dtype='float32')
        degrees = np.expand_dims(degrees, -1)

        src_ids, tgt_ids = RandomGraphs.gen_vertex_pairs(n=n, size=5 * n)
        edge_index = np.array(graph.to_directed(as_view=True).edges).T
        res = Data(x=torch.from_numpy(degrees), y=torch.from_numpy(betweenness),
                   src_ids=torch.from_numpy(src_ids), tgt_ids=torch.from_numpy(tgt_ids),
                   edge_index=torch.from_numpy(edge_index), num_nodes=n)
        return res


class GraphDataModule(LightningDataModule):
    train_dataset: RandomGraphs
    valid_dataset: RandomGraphs
    next_train_worker: ThreadWithReturnValue[RandomGraphs]
    next_valid_worker: ThreadWithReturnValue[RandomGraphs]

    def __init__(self,
                 min_nodes: int, max_nodes: int, nb_train_graphs: int, nb_valid_graphs: int,
                 batch_size: int, graph_type: str = 'powerlaw', repeats: int = 8, regenerate_epoch_interval: int = 5,
                 cache_dir: Union[str, os.PathLike, None] = None,
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
        self.cache_dir: Path = Path(cache_dir)
        self.train_epochs: int = 0
        self.valid_epochs: int = 0
        self.workers: int = max(os.cpu_count() - 1, 1)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    @staticmethod
    def create_dataset(min_nodes: int, max_nodes: int, graph_type: str,
                       nb_graphs: int, repeats: int, save_path: Path) -> RandomGraphs:
        return RandomGraphs(min_nodes=min_nodes, max_nodes=max_nodes, graph_type=graph_type,
                            nb_graphs=nb_graphs, repeats=repeats, save_path=save_path)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """
        Generate random graphs and return the loader for those.
        If it's the initial epoch start the generation on the main thread.
        Otherwise start the generation on s separate one so that the training/main thread is not blocked
        """
        dataset_path = self.cache_dir / f'train_ep{self.train_epochs}_' \
                                        f'{self.min_nodes}_{self.max_nodes}_{self.graph_type}.data'
        if self.train_epochs % self.regenerate_every_epochs == 0:
            print(f'Generating {self.nb_train_graphs} new Train graphs - [{self.min_nodes}-{self.max_nodes}]...')
            self.train_dataset = self.next_train_worker.join() if self.train_epochs != 0 else RandomGraphs(
                min_nodes=self.min_nodes, max_nodes=self.max_nodes, graph_type=self.graph_type,
                nb_graphs=self.nb_train_graphs, repeats=self.repeats, save_path=dataset_path,
            )
            self.next_train_worker = ThreadWithReturnValue(target=GraphDataModule.create_dataset,
                                                           args=(self.min_nodes, self.max_nodes, self.graph_type,
                                                                 self.nb_train_graphs, self.repeats, dataset_path))
            self.next_train_worker.start()

        self.train_epochs += 1
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        """
        Generate random graphs and return the loader for those.
        If it's the initial epoch start the generation on the main thread.
        Otherwise start the generation on s separate one so that the training/main thread is not blocked
        """
        dataset_path = self.cache_dir / f'valid_ep{self.valid_epochs}_' \
                                        f'{self.min_nodes}_{self.max_nodes}_{self.graph_type}.data'
        if self.valid_epochs % self.regenerate_every_epochs == 0:
            print(f'Generating {self.nb_valid_graphs} new Validation graphs - [{self.min_nodes}-{self.max_nodes}]...')
            self.valid_dataset = self.next_valid_worker.join() if self.valid_epochs != 0 else RandomGraphs(
                min_nodes=self.min_nodes, max_nodes=self.max_nodes, graph_type=self.graph_type,
                nb_graphs=self.nb_valid_graphs, repeats=1, save_path=dataset_path,
            )
            self.next_valid_worker = ThreadWithReturnValue(target=GraphDataModule.create_dataset,
                                                           args=(self.min_nodes, self.max_nodes, self.graph_type,
                                                                 self.nb_valid_graphs, 1, dataset_path))
            self.next_valid_worker.start()

        self.valid_epochs += 1
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        batch.to(device)
        return batch
