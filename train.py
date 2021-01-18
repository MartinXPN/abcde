import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE, DrBC
from abcde.util import fix_random_seed, ExperimentSetup


if __name__ == '__main__':
    # Fix the seed for reproducibility
    fix_random_seed(42)
    experiment = ExperimentSetup(name='drbc', create_latest=True, long_description="""
    Vanilla DrBC
    """)

    model = DrBC(nb_gcn_cycles=5, lr_reduce_patience=2)
    data = GraphDataModule(min_nodes=400, max_nodes=500, nb_train_graphs=100, nb_valid_graphs=100,
                           batch_size=16, graph_type='powerlaw', regenerate_epoch_interval=5,
                           repeats=8, verbose=False)
    trainer = pl.Trainer(logger=[
                            CSVLogger(experiment.log_dir, name='history'),
                            TensorBoardLogger(experiment.log_dir, name=experiment.name, default_hp_metric=False),
                            # AimLogger(experiment=experiment.name)
                         ],
                         gpus=-1 if torch.cuda.is_available() else None, auto_select_gpus=True, log_gpu_memory='all',
                         max_epochs=100, terminate_on_nan=True,
                         enable_pl_optimizer=True, reload_dataloaders_every_epoch=True,
                         callbacks=[
                             EarlyStopping(monitor='val_kendal', patience=5, verbose=True, mode='max'),
                             ModelCheckpoint(dirpath=experiment.model_save_path, filename='drbc-{epoch:02d}',
                                             monitor='val_kendal', save_top_k=5, verbose=True, mode='max'),
                             LearningRateMonitor(logging_interval='step'),
                         ])
    trainer.fit(model, datamodule=data)
