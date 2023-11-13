import csv

from lightning.pytorch.callbacks import BasePredictionWriter
import torch
from tqdm import tqdm


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_path: str, write_interval):
        super().__init__(write_interval)
        self.output_path = output_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        with open(self.output_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            pred = [0] * 91
            pred[0] = 4400
            for batch, prediction in tqdm(zip(batch_indices[0], predictions)):
                for batch_idx, prediction_idx in zip(batch, prediction):
                    window = torch.argmax(prediction_idx).cpu().item()
                    pred[3 + batch_idx % 79] = window
                    if batch_idx % 79 == 78:
                        csv_writer.writerow(pred)
                        pred = [0] * 91
                        pred[0] = 4401 + batch_idx // 79
