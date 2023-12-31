from model import CDIL, AutoEncoder
from dataloader import ECGDataModule, HDF5ECGDataset
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping


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
    model = AutoEncoder(cdil, lr=config['learning_rate'], wd=config['weight_decay'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='cdil-{epoch:02d}-hc' + str(config['hidden_channels']) + '-nl' + str(config['num_layers']) + '-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    wandb_logger = WandbLogger(project='hackhealth2023-autoencoder', config=config)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        enable_progress_bar=True,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':    
    # setup argparse to load smoke test boolean flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    
    # add search_space parameters
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    if args.smoke_test:
        max_epochs = 1
        num_samples = 1
    else:
        max_epochs = 100
        num_samples = 100
    
    search_space = {
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "kernel_size": args.kernel_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    train_func(search_space, max_epochs)