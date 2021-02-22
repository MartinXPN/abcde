from pathlib import Path

import torch
from aim.sdk.adapters.pytorch_lightning import AimLogger
from knockknock import telegram_sender
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed, ExperimentSetup


# Fix the seed for reproducibility
fix_random_seed(42)
experiment = ExperimentSetup(name='grad-clip', create_latest=True, long_description="""
Try dropping edges while training
Graphs are only of 'powerlaw' type.
Use unique convolutions.
Use blocks of convolutions followed with max pooling and skip connections
Use gradient clipping
""")
torch.multiprocessing.set_sharing_strategy('file_system')


def fit(t: Trainer):
    t.fit(model, datamodule=data)
    return t.callback_metrics


if __name__ == '__main__':
    loggers = [
        CSVLogger(experiment.log_dir, name='history'),
        TensorBoardLogger(experiment.log_dir, name=experiment.name, default_hp_metric=False),
        WandbLogger(name=experiment.name, save_dir=experiment.log_dir, project='abcde', save_code=True, notes=experiment.long_description),
        AimLogger(experiment=experiment.name),
    ]
    # Previous best: nb_gcn_cycles=(4, 4, 6, 6, 8), conv_sizes=(64, 64, 32, 32, 16), drops=(0, 0, 0, 0, 0)
    model = ABCDE(nb_gcn_cycles=(4, 4, 6, 6, 8, 8),
                  conv_sizes=(48, 48, 32, 32, 24, 24),
                  drops=(0.3, 0.3, 0.2, 0.2, 0.1, 0.1),
                  lr_reduce_patience=2, dropout=0.1)
    data = GraphDataModule(min_nodes=4000, max_nodes=5000, nb_train_graphs=160, nb_valid_graphs=240,
                           batch_size=16, graph_type='powerlaw', repeats=8, regenerate_epoch_interval=10,
                           cache_dir=Path('datasets') / 'cache')
    trainer = Trainer(logger=loggers, gradient_clip_val=0.1,
                      gpus=-1 if torch.cuda.is_available() else None, auto_select_gpus=True,
                      max_epochs=50, terminate_on_nan=True, enable_pl_optimizer=True,
                      reload_dataloaders_every_epoch=True,
                      callbacks=[
                          EarlyStopping(monitor='val_kendal', patience=6, verbose=True, mode='max'),
                          ModelCheckpoint(dirpath=experiment.model_save_path, filename='drop-{epoch:02d}-{val_kendal:.2f}', monitor='val_kendal', save_top_k=5, verbose=True, mode='max'),
                          LearningRateMonitor(logging_interval='epoch'),
                      ])
    fit(trainer)
