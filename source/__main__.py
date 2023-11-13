from lightning.pytorch.cli import LightningCLI
from loguru import logger
import torch
from tqdm import tqdm


def cli():
    cli = LightningCLI()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    logger.info("********** START SLEEP APNEA PROGRAM **********")
    cli()
    logger.info("********** END SLEEP APNEA PROGRAM **********")
