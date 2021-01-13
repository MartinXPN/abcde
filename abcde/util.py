import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def fix_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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
