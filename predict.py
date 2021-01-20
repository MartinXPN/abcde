import time
from pprint import pprint
from typing import Union, IO

import fire
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from abcde.models import ABCDE, DrBC
from abcde.util import display_help_stdout


def predict(model_path: Union[str, IO],
            data_test:  Union[str, IO],
            label_file: Union[str, IO]):
    model = DrBC.load_from_checkpoint(model_path)
    model.eval()
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
    res['run_time'] = end - start,
    pprint(res, sort_dicts=False)


if __name__ == '__main__':
    fire.core.Display = display_help_stdout
    fire.Fire(predict)
