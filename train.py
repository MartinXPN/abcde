import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed, ExperimentSetup


eval_interval = 8       # Evaluate the model once every n epochs
fix_random_seed(42)     # Fix the seed for reproducibility
experiment = ExperimentSetup(name='gcn_conv_8_drop_edge', create_latest=True, long_description="""
Use 8 unique GCNConvolutions for training
Use DropEdge on undirected input graph with dropout rate of 0.3
""")

model = ABCDE(nb_gcn_cycles=8, lr_reduce_patience=3 * eval_interval)
data = GraphDataModule(min_nodes=400, max_nodes=500, nb_train_graphs=160, nb_valid_graphs=320,
                       batch_size=16, graph_type='powerlaw', regenerate_epoch_interval=5 * eval_interval,
                       verbose=False)
trainer = pl.Trainer(logger=[
                        CSVLogger(save_dir=experiment.log_dir, name='history'),
                        TensorBoardLogger(save_dir=experiment.log_dir, name=experiment.name, default_hp_metric=False),
                        # AimLogger(experiment=experiment.name)
                     ],
                     gpus=-1 if torch.cuda.is_available() else None, auto_select_gpus=True, log_gpu_memory='all',
                     max_epochs=10 * eval_interval, terminate_on_nan=True,
                     enable_pl_optimizer=True, reload_dataloaders_every_epoch=True,
                     check_val_every_n_epoch=eval_interval,
                     progress_bar_refresh_rate=30,
                     callbacks=[
                         EarlyStopping(monitor='val_kendal', patience=5 * eval_interval, verbose=True, mode='max'),
                         ModelCheckpoint(dirpath=experiment.model_save_path, filename='best',
                                         monitor='val_kendal', save_top_k=1, verbose=True, mode='max'),
                         LearningRateMonitor(logging_interval='step'),
                     ])
trainer.fit(model, datamodule=data)
