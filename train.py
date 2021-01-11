import pytorch_lightning as pl
from aim.sdk.adapters.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed, ExperimentSetup


eval_interval = 8       # Evaluate the model once every n epochs
fix_random_seed(42)     # Fix the seed for reproducibility
experiment = ExperimentSetup(name='abcde_unique_convs', create_latest=True, long_description="""
Skip connection from the input node feature of 32 length to the final Fully Connected layers
Setting all the convolutions in the model to be unique instead of reusing the same conv weights
GRU cell is still reused on each step
""")

model = ABCDE(nb_gcn_cycles=5, lr_reduce_patience=3 * eval_interval)
data = GraphDataModule(min_nodes=400, max_nodes=500, nb_train_graphs=160, nb_valid_graphs=320,
                       batch_size=16, graph_type='powerlaw', regenerate_epoch_interval=5 * eval_interval,
                       verbose=False)
trainer = pl.Trainer(logger=[
                        TensorBoardLogger(save_dir=experiment.log_dir, name=experiment.name, default_hp_metric=False),
                        AimLogger(experiment=experiment.name)
                     ],
                     auto_select_gpus=True, max_epochs=10 * eval_interval, terminate_on_nan=True,
                     enable_pl_optimizer=True, reload_dataloaders_every_epoch=True,
                     check_val_every_n_epoch=eval_interval,
                     callbacks=[
                         EarlyStopping(monitor='val_kendal', patience=5 * eval_interval, verbose=True, mode='max'),
                         ModelCheckpoint(dirpath=experiment.model_save_path, filename='best',
                                         monitor='val_kendal', save_top_k=1, verbose=True, mode='max'),
                         LearningRateMonitor(logging_interval='step'),
                     ])
trainer.fit(model, datamodule=data)
