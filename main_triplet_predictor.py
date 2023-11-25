from model import CDIL, AutoEncoder, Predictor, TripletPredictor
from dataloader import ECGDataModule, HDF5ECGDataset
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from resnet_model import ResNet
from tsai.models.TSPerceiver import TSPerceiver


def train_func(config, max_epochs, num_samples):
    dm = ECGDataModule('dataset/', batch_size=config['batch_size'],
                       mode=HDF5ECGDataset.Mode.MODE_HH_TRIPLETS,  # Use the mode for triplets
                       num_workers=8, sample_size=num_samples,
                       train_fraction=0.8, dev_fraction=0.2, test_fraction=0.0)


    core = ResNet(normalize=True, propagate_normalization=False, embedding_size=config['output_channels'],
                      dropout=config['dropout'])


    model = TripletPredictor(core, lr=config['learning_rate'], wd=config['weight_decay'])


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename=str(config['model']) + '-{epoch:02d}-hc' + str(config['hidden_channels']) + '-nl' + str(
            config['num_layers']) + '-oc' + str(config['output_channels']) + '-{val_accuracy:.2f}',
        save_top_k=1,
        mode='max',
    )
    wandb_logger = WandbLogger(project='hackhealth2023-triplet-predictor', config=config)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
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
    # parser.add_argument("--model", type=str, required=True, choices=['resnet'], help='model to use')
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--output_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()


    max_epochs = 100
    num_samples = 20000
    batch_size = args.batch_size

    search_space = {
        "model":"resnet",  # "cdil" or "resnet
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