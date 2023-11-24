from model import CDIL, AutoEncoder, Predictor
from dataloader import ECGDataModule, HDF5ECGDataset
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from resnet_model import ResNet


def train_func(config, max_epochs, num_samples):
    dm = ECGDataModule('datasets/hhmusedata', batch_size=config['batch_size'], mode=HDF5ECGDataset.Mode.MODE_HH_CLASSIFIER_SIMPLE,
                       num_workers=7, sample_size=num_samples)
    
    if config['model'] == 'cdil':
        core = CDIL(
            input_channels=12,
            hidden_channels=config['hidden_channels'],
            output_channels=config['output_channels'],
            num_layers=config['num_layers'],
            kernel_size=config['kernel_size']
        )
    elif config['model'] == 'resnet':
        core = ResNet(normalize=True, propagate_normalization=False, embedding_size=config['output_channels'], dropout=config['dropout'])

    model = Predictor(core, config['model'], config['output_channels'], lr=config['learning_rate'], wd=config['weight_decay'], dropout=config['dropout'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='checkpoints/',
        filename=str(config['model']) + '-{epoch:02d}-hc' + str(config['hidden_channels']) + '-nl' + str(config['num_layers']) + '-oc' + str(config['output_channels']) +'-{val_accuracy:.2f}',
        save_top_k=1,
        mode='max',
    )
    wandb_logger = WandbLogger(project='hackhealth2023-predictor', config=config)
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='max'
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
    parser.add_argument("--model", type=str, required=True, choices=['cdil', 'resnet'], help='model to use')
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--output_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    args = parser.parse_args()
    
    if args.smoke_test:
        max_epochs = 1
        num_samples = 500
        batch_size = 32
    else:
        max_epochs = 100
        num_samples = 20000
        batch_size = args.batch_size
    
    search_space = {
        "model": args.model, # "cdil" or "resnet
        "hidden_channels": args.hidden_channels,
        "output_channels": args.output_channels,
        "num_layers": args.num_layers,
        "kernel_size": args.kernel_size,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
    }
    train_func(search_space, max_epochs, num_samples)