from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger

from abcde.data import GraphDataModule
from abcde.models import ABCDE
from abcde.util import fix_random_seed, ExperimentSetup


# Fix the seed for reproducibility
fix_random_seed(42)
experiment = ExperimentSetup(name='vanilla_abcde', create_latest=True, long_description="""
Use PReLU activation
Use Adam optimizer with big learning rate
Try to have variable number of edges in the generated graphs
Try dropping edges while training
Graphs are only of 'powerlaw' type.
Use unique convolutions.
Use blocks of convolutions followed with max pooling and skip connections
Use gradient clipping
""")
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    loggers = [
        CSVLogger(experiment.log_dir, name='history'),
        TensorBoardLogger(experiment.log_dir, name=experiment.name, default_hp_metric=False),
        WandbLogger(name=experiment.name, save_dir=experiment.log_dir, project='abcde', save_code=True, notes=experiment.long_description),
        # AimLogger(experiment=experiment.name),
    ]
    model = ABCDE(nb_gcn_cycles=(4, 4, 6, 6, 8, 8),
                  conv_sizes=(48, 48, 32, 32, 24, 24),
                  drops=(0.3, 0.3, 0.2, 0.2, 0.1, 0.1),
                  lr_reduce_patience=2, dropout=0.1)
    data = GraphDataModule(min_nodes=4000, max_nodes=5000, nb_train_graphs=160, nb_valid_graphs=240,
                           batch_size=16, graph_type='powerlaw', repeats=8, regenerate_epoch_interval=10,
                           cache_dir=Path('datasets') / 'cache')
    trainer = Trainer(logger=loggers, gradient_clip_val=1,
                      gpus=-1 if torch.cuda.is_available() else None, auto_select_gpus=True,
                      max_epochs=50, terminate_on_nan=True, reload_dataloaders_every_epoch=True,
                      callbacks=[
                          EarlyStopping(monitor='val_kendal', patience=5, verbose=True, mode='max'),
                          ModelCheckpoint(dirpath=experiment.model_save_path, filename='model-{epoch:02d}-{val_kendal:.2f}', monitor='val_kendal', save_top_k=5, verbose=True, mode='max'),
                          LearningRateMonitor(logging_interval='epoch'),
                      ])
    trainer.fit(model, datamodule=data)
    print(trainer.callback_metrics)
