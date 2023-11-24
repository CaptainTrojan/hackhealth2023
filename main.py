from model import CDIL, AutoEncoder
from dataloader import ECGDataModule, HDF5ECGDataset
import argparse
import pytorch_lightning as pl


def train_func(config):
    dm = ECGDataModule('datasets/ikem', batch_size=config['batch_size'], mode=HDF5ECGDataset.Mode.MODE_MASKED_AUTOENCODER,
                       num_workers=7)
    cdil = CDIL(
        input_channels=12,
        hidden_channels=config['hidden_channels'],
        output_channels=12,
        num_layers=config['num_layers'],
        kernel_size=config['kernel_size']
    )
    model = AutoEncoder(cdil)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=False
    )
    trainer.fit(model, datamodule=dm)
    
    return ...


if __name__ == '__main__':    
    # setup argparse to load ray address and smoke test boolean flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    if args.smoke_test:
        max_epochs = 1
        num_samples = 1
    else:
        max_epochs = 100
        num_samples = 100
    
    search_space = {
        "hidden_channels": 32,
        "num_layers": 4,
        "kernel_size": 3,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
    }
    train_func(search_space)