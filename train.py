from pathlib import Path

import torch
from aim.sdk.adapters.pytorch_lightning import AimLogger
from knockknock import telegram_sender
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed, ExperimentSetup


# Fix the seed for reproducibility
fix_random_seed(42)
experiment = ExperimentSetup(name='drop', create_latest=True, long_description="""
Try dropping edges while training
Graphs are only of 'powerlaw' type.
Use unique convolutions.
Use blocks of convolutions followed with max pooling and skip connections
""")
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    loggers = [
        CSVLogger(experiment.log_dir, name='history'),
        TensorBoardLogger(experiment.log_dir, name=experiment.name, default_hp_metric=False),
        AimLogger(experiment=experiment.name),
    ]
    # Previous best: nb_gcn_cycles=(4, 4, 6, 6, 8), conv_sizes=(64, 64, 32, 32, 16), drops=(0, 0, 0, 0, 0)
    model = ABCDE(nb_gcn_cycles=(4, 4, 6, 6, 8, 8),
                  conv_sizes=(64, 48, 32, 32, 24, 24),
                  drops=(0.4, 0.3, 0.2, 0.2, 0.1, 0.1),
                  lr_reduce_patience=2, dropout=0.1)
    data = GraphDataModule(min_nodes=4000, max_nodes=5000, nb_train_graphs=160, nb_valid_graphs=240,
                           batch_size=16, graph_type='powerlaw', repeats=8, regenerate_epoch_interval=10,
                           cache_dir=Path('datasets') / 'cache')
    trainer = Trainer(logger=loggers,
                      gpus=-1 if torch.cuda.is_available() else None, auto_select_gpus=True,
                      max_epochs=100, terminate_on_nan=True, enable_pl_optimizer=True,
                      reload_dataloaders_every_epoch=True,
                      callbacks=[
                          EarlyStopping(monitor='val_kendal', patience=5, verbose=True, mode='max'),
                          ModelCheckpoint(dirpath=experiment.model_save_path, filename='drop-{epoch:02d}-{val_kendal:.2f}', monitor='val_kendal', save_top_k=5, verbose=True, mode='max'),
                          LearningRateMonitor(logging_interval='epoch'),
                      ])
    trainer.fit = telegram_sender(token='1653878275:AAEIr-mLt9-SSAyYPon1n-CgFQpINjUWHDw', chat_id=695404691)(trainer.fit)
    trainer.fit(model, datamodule=data)
