import os
import h5py
from lightning.pytorch.core.datamodule import DataLoader, Dataset, LightningDataModule
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


class SleepApneaDataset(Dataset):
    def __init__(self, data_dir: str, download: bool = False, predict: bool = False):
        super().__init__()
        self.data_dir = data_dir

        self.x = None
        self.y = None

        self.init(download, predict)

    def __getitem__(self, item):
        patient_id = item // 79
        window_id = item % 79
        if self.y is None:
            return (
                self.x[patient_id, :, window_id * 100: window_id * 100 + 1100].unsqueeze(-2),
                0
            )
        return (
            self.x[patient_id, :, window_id * 100: window_id * 100 + 1100].unsqueeze(-2),
            self.y[patient_id, window_id + 2]
        )

    def __len__(self):
        # 79 windows per patient (11-second windows)
        # 4400 patients
        return self.x.size(0) * 79

    def init(self, download: bool, predict: bool):
        if download:
            return

        if not predict:
            filename_x_train = os.path.join(self.data_dir, "X_train.h5")
            filename_y_train = os.path.join(self.data_dir, "y_train.csv")

            x_file = h5py.File(filename_x_train, 'r')
            dset_x = x_file['data']
            # x shape [4_400, 72_000]
            self.x = torch.from_numpy(
                pd.DataFrame(np.array(dset_x))
                .drop(columns=[0, 1], axis=1)
                .values).reshape(4400, 8, 9000).to(torch.float)

            dset_y = pd.read_csv(filename_y_train)
            # y shape [4_400, 90]
            self.y = torch.from_numpy(pd.DataFrame(dset_y).drop(columns=["ID"]).values)
        else:
            filename_x_train = os.path.join(self.data_dir, "X_test.h5")
            x_file = h5py.File(filename_x_train, 'r')
            dset_x = x_file['data']
            # x shape [4_400, 72_000]
            self.x = torch.from_numpy(
                pd.DataFrame(np.array(dset_x))
                .drop(columns=[0, 1], axis=1)
                .values).reshape(4400, 8, 9000).to(torch.float)


class SleepApnea(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_dir: str,
            num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        SleepApneaDataset(data_dir=self.hparams.data_dir, download=True)

    def setup(self, stage):
        match stage:
            case "fit":
                self.full_dataset = SleepApneaDataset(data_dir=self.hparams.data_dir)

                train_indices = list(range(200 * 4 * 79, 4400 * 79))
                val_indices = list(range(0, 200 * 4 * 79))
                self.train_dataset = Subset(self.full_dataset, train_indices)
                self.val_dataset = Subset(self.full_dataset, val_indices)

            case "predict":
                self.predict_dataset = SleepApneaDataset(
                    data_dir=self.hparams.data_dir,
                    predict=True
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
