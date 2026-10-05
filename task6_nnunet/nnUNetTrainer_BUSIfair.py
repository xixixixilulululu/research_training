import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_BUSIfair(nnUNetTrainer):
    """nnU-Net defaults, but training length and seed matched to the BUSI comparison."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.num_iterations_per_epoch = 33
        self.num_val_iterations_per_epoch = 10
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
