from datetime import datetime
from pathlib import Path

import fire
import pytorch_lightning as pl
from aim import Session
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from data import GraphDataModule
from models import ABCDE
from util import fix_random_seed


class Gym:
    model: pl.LightningModule
    data: GraphDataModule

    def __init__(self, experiment: str = 'vanilla_drbc'):
        """
        Gym keeps track of the model and the data on which it is trained
        It handles the logging, experiment tracking, data generation, initialization, and training
        @param experiment: description of the experiment
        """
        self.experiment_path = Path('experiments') / datetime.now().replace(microsecond=0).isoformat()
        self.model_save_path = self.experiment_path / 'models/'
        self.log_dir = self.experiment_path / 'logs/'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        latest = Path('experiments/latest/').absolute()
        latest.unlink(missing_ok=True)
        latest.symlink_to(self.experiment_path.absolute(), target_is_directory=True)

        print(f'Logging experiments at: `{self.experiment_path.absolute()}`')
        self.aim_session = Session(experiment=experiment)

    def train(self) -> pl.Trainer:
        self.model = ABCDE()
        self.data = GraphDataModule(min_nodes=400, max_nodes=500, nb_train_graphs=500, nb_valid_graphs=100,
                                    batch_size=16, graph_type='powerlaw', regenerate_every_epochs=5)

        trainer = pl.Trainer(auto_select_gpus=True, max_epochs=100, terminate_on_nan=True, enable_pl_optimizer=True,
                             reload_dataloaders_every_epoch=True, callbacks=[
                                 EarlyStopping(monitor='val_kendal', patience=5, verbose=True, mode='max'),
                                 ModelCheckpoint(dirpath=self.model_save_path, filename='best', monitor='val_kendal',
                                                 save_top_k=1, verbose=True, mode='max'),
                                 LearningRateMonitor(logging_interval='step'),
                             ])

        trainer.fit(self.model, datamodule=self.data)
        return trainer


if __name__ == '__main__':
    fix_random_seed(42)
    fire.Fire(Gym)
