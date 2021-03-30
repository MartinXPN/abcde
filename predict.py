import glob
import time
from pathlib import Path
from pprint import pprint
from typing import Union, IO

import fire
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from abcde.models import ABCDE
from abcde.util import display_help_stdout


def synthetic(model_path: Union[str, IO],
              directory: Union[str, IO]):
    model = ABCDE.load_from_checkpoint(model_path)
    model = model.eval()
    model.freeze()

    directory = str(directory) + '/*_score.txt'
    statistics = []
    for test_number in glob.glob(directory):

        label = pd.read_csv(test_number, delim_whitespace=True, header=None, names=['node_id', 'betweenness'])
        assert np.array_equal(label.node_id.map(lambda x: int(x)).values, np.arange(len(label)))
        label = np.expand_dims(label.betweenness.values, -1)

        g = nx.read_weighted_edgelist(test_number.replace('_score.txt', '.txt'), nodetype=int)
        edge_index = np.array(g.to_directed(as_view=True).edges).T

        degrees = nx.degree_centrality(g)
        degrees = np.array([degrees[n] for n in range(len(label))], dtype='float32')
        degrees = np.expand_dims(degrees, -1)

        start = time.time()
        graph = Data(x=torch.from_numpy(degrees),
                     y=torch.from_numpy(label),
                     edge_index=torch.from_numpy(edge_index))

        res = model.validation_step(graph, batch_idx=0)[0]
        end = time.time()
        res['run_time'] = end - start
        statistics.append(res)

    statistics = pd.DataFrame(statistics)
    # print(statistics)
    print(statistics.describe())


def real(model_path: Union[str, IO],
         data_test:  Union[str, IO],
         label_file: Union[str, IO]):
    model = ABCDE.load_from_checkpoint(model_path)
    model = model.eval()
    model.freeze()

    label = pd.read_csv(label_file, delim_whitespace=True, header=None, names=['node_id', 'betweenness'])
    assert np.array_equal(label.node_id.map(lambda x: int(x.replace(':', ''))).values, np.arange(len(label)))
    label = np.expand_dims(label.betweenness.values, -1)

    g = nx.read_weighted_edgelist(data_test, nodetype=int)
    edge_index = np.array(g.to_directed(as_view=True).edges).T

    degrees = nx.degree_centrality(g)
    degrees = np.array([degrees[n] for n in range(len(label))], dtype='float32')
    degrees = np.expand_dims(degrees, -1)

    start = time.time()
    graph = Data(x=torch.from_numpy(degrees),
                 y=torch.from_numpy(label),
                 edge_index=torch.from_numpy(edge_index))
    print('Graph:', graph)

    res = model.validation_step(graph, batch_idx=0)[0]
    end = time.time()
    res['run_time'] = end - start
    pprint(res, sort_dicts=False)


def evaluate_all(model_path: Union[str, IO],
                 datasets_dir: Union[str, IO]):
    # Evaluate on real datasets
    real_dir = Path(datasets_dir) / 'real'
    for dataset in ['com-youtube', 'amazon', 'cit-Patents', 'dblp', 'com-lj']:
        print('Evaluating the dataset:', dataset)
        real(model_path=model_path,
             data_test=real_dir / (dataset + '.txt'),
             label_file=real_dir / (dataset + '_score.txt'))

    # Evaluate on synthetic datasets
    synth_dir = Path(datasets_dir) / 'synthetic'
    for dataset in ['5000', '10000', '20000', '50000', '100000']:
        print('Evaluating the dataset:', dataset)
        synthetic(model_path=model_path, directory=synth_dir / dataset)


if __name__ == '__main__':
    fire.core.Display = display_help_stdout
    fire.Fire({
        'real': real,
        'synthetic': synthetic,
        'all': evaluate_all,
    })
