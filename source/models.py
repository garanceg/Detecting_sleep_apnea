import lightning
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torch import nn
import torchmetrics


class ConvModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 3, kernel_size=100, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(3, 50, kernel_size=10),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(50, 30, kernel_size=30),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.BatchNorm1d(30),
            nn.Flatten(),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        # input shape [bs, 1, 1100]
        # output shape [bs, 1350]
        return self.model(x)


class SleepApneaModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.models = []
        for i in range(8):
            self.models.append(ConvModule().cuda())

        self.fc = nn.Sequential(
            nn.Linear(10_800, 2_048),
            nn.ReLU(),
            nn.Linear(2_048, 2),
        )

    def forward(self, x):
        # input [bs, 8, 1, 1100]
        # out [bs, 10_800]
        out = torch.cat(list(self.models[i](x[:, i, :, :]) for i in range(8)), dim=1)
        # output [bs, 2]
        return self.fc(out)


class SleepApnea(lightning.pytorch.core.module.LightningModule):
    def __init__(self):
        super().__init__()

        num_classes = 2
        self.save_hyperparameters()
        self.model = SleepApneaModule()

        self.loss = nn.CrossEntropyLoss()

        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes, average="micro"
                ),
                torchmetrics.Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                torchmetrics.Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                torchmetrics.F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            ]
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

        self.train_confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
            normalize="true",
        )
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
            normalize="true",
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch, batch_idx, step: str):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        getattr(self, f"{step}_metrics").update(y_hat, y)
        getattr(self, f"{step}_confusion_matrix").update(y_hat, y)

        self.log_dict(getattr(self, f"{step}_metrics"), on_step=False, on_epoch=True)
        self.log(f"{step}_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def on_validation_epoch_end(self):
        self._shared_epoch_end("val")

    def on_train_epoch_end(self):
        self._shared_epoch_end("train")

    def _shared_epoch_end(self, step: str):
        array = np.around(
            getattr(self, f"{step}_confusion_matrix").compute().cpu().numpy(),
            decimals=2,
        )
        cm = ConfusionMatrixDisplay(array)
        cm.plot()
        cm.plot(cmap=sn.color_palette("rocket", as_cmap=True))

        tb_logger = self.trainer.loggers[0]
        tb_logger.experiment.add_figure(f"{step}_ConfMat", cm.figure_, self.current_epoch)

        getattr(self, f"{step}_metrics").reset()
        getattr(self, f"{step}_confusion_matrix").reset()
        plt.close()
