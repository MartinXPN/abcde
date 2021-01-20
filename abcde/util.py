import os
import random
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional, TypeVar, Generic

import numpy as np
import torch


def fix_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def display_help_stdout(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)


class ExperimentSetup:
    def __init__(self, name: str, long_description: Optional[str] = None, create_latest: bool = False):
        """ Keeps track of the experiment path, model save path, log directory, and sessions """
        self.name = name
        self.long_description = long_description
        self.experiment_time = datetime.now().replace(microsecond=0).isoformat()

        self.experiment_path = Path('experiments') / f'{self.experiment_time}_{self.name}'
        self.model_save_path = self.experiment_path / 'models/'
        self.log_dir = self.experiment_path / 'logs/'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if long_description:
            with open(self.log_dir / 'description.txt', 'w') as f:
                f.write(long_description.strip())

        if create_latest:
            latest = Path('experiments/latest/').absolute()
            latest.unlink(missing_ok=True)
            latest.symlink_to(self.experiment_path.absolute(), target_is_directory=True)

        print(f'Logging experiments at: `{self.experiment_path.absolute()}`')


T = TypeVar('T')


class ThreadWithReturnValue(Thread, Generic[T]):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._return: Optional[T] = None

    # noinspection PyUnresolvedReferences
    def run(self):
        try:
            if self._target:
                self._return = self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def join(self, timeout: Optional[float] = None) -> T:
        super().join(timeout)
        return self._return
