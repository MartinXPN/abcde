import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed, ExperimentSetup


fix_random_seed(42)
experiment_setup = ExperimentSetup(experiment='vanilla_drbc', create_latest=True)
model = ABCDE()
data = GraphDataModule(min_nodes=400, max_nodes=500, nb_train_graphs=80, nb_valid_graphs=20,
                       batch_size=16, graph_type='powerlaw', regenerate_epoch_interval=5)
trainer = pl.Trainer(auto_select_gpus=True, max_epochs=100, terminate_on_nan=True, enable_pl_optimizer=True,
                     reload_dataloaders_every_epoch=True,
                     callbacks=[
                         EarlyStopping(monitor='val_kendal', patience=5, verbose=True, mode='max'),
                         ModelCheckpoint(dirpath=experiment_setup.model_save_path, filename='best',
                                         monitor='val_kendal', save_top_k=1, verbose=True, mode='max'),
                         LearningRateMonitor(logging_interval='step'),
                     ])
trainer.fit(model, datamodule=data)
