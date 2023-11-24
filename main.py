from model import CDIL, AutoEncoder
from dataloader import ECGDataModule, HDF5ECGDataset
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def train_func(config, max_epochs):
    dm = ECGDataModule('datasets/ikem', batch_size=config['batch_size'], mode=HDF5ECGDataset.Mode.MODE_MASKED_AUTOENCODER,
                       num_workers=7, sample_size=20000)
    cdil = CDIL(
        input_channels=12,
        hidden_channels=config['hidden_channels'],
        output_channels=12,
        num_layers=config['num_layers'],
        kernel_size=config['kernel_size']
    )
    model = AutoEncoder(cdil)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='cdil-{epoch:02d}-hc' + str(config['hidden_channels']) + '-nl' + str(config['num_layers']) + '-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    wandb_logger = WandbLogger(project='hackhealth2023-autoencoder', config=config)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=True,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':    
    # setup argparse to load smoke test boolean flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    
    # add search_space parameters
    
    args = parser.parse_args()
    
    if args.smoke_test:
        max_epochs = 1
        num_samples = 1
    else:
        max_epochs = 100
        num_samples = 100
    
    search_space = {
        "hidden_channels": 64,
        "num_layers": 5,
        "kernel_size": 3,
        "batch_size": 256,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
    }
    train_func(search_space, max_epochs)