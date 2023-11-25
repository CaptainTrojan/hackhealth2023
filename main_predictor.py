from model import CDIL, AutoEncoder, Predictor
from dataloader import ECGDataModule, HDF5ECGDataset
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from resnet_model import ResNet
from tsai.models.TSPerceiver import TSPerceiver


def train_func(config, max_epochs, num_samples):
    dm = ECGDataModule('datasets/hhmusedata', batch_size=config['batch_size'], mode=HDF5ECGDataset.Mode.MODE_HH_CLASSIFIER_SIMPLE,
                       num_workers=7, sample_size=num_samples, train_fraction=0.8, dev_fraction=0.2, test_fraction=0.0)
    
    core = prepare_core(config)
    model = Predictor(core, config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        dirpath='checkpoints/',
        filename=stringify_config(config) + '-{val_accuracy:.2f}',
        save_top_k=1,
        mode='max',
    )
    wandb_logger = WandbLogger(project='hackhealth2023-predictor', config=config)
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00,
        patience=30,
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
    
def stringify_config(config):
    ret = ""
    for key in config:
        shortened_key = "".join(v[0] for v in key.split("_"))
        value = config[key]
        if type(value) == float:
            value = round(value, 3)
        if type(value) == bool:
            if value:
                value = "T"
            else:  
                value = "F"
        ret += f"{shortened_key}{value}-"
    return ret

def prepare_core(config):
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
    elif config['model'] == 'tsai01':
        core = TSPerceiver(c_in=12, c_out=config['output_channels'], seq_len=4096, 
                           attn_dropout=config['dropout'], fc_dropout=config['dropout'],
                           n_layers=config['num_layers'])
                           
    return core

if __name__ == '__main__':    
    # setup argparse to load smoke test boolean flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    
    # add search_space parameters
    parser.add_argument("--model", type=str, default='cdil', choices=['cdil', 'resnet', 'tsai01'], help='model to use')
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--output_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no_ecg", action="store_true")
    parser.add_argument("--added_feats", action="store_true")
    parser.add_argument("--af_onehot", action="store_true")
    
    parser.add_argument("--checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.smoke_test:
        max_epochs = 1
        num_samples = 500
        batch_size = 32
    else:
        max_epochs = 1000
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
        "ignore_ecg": args.no_ecg,
        "added_features": args.added_feats,
        "added_features_onehot": args.af_onehot
    }

    if args.checkpoint is not None:
        core = prepare_core(search_space)
        
        model = Predictor.load_from_checkpoint(args.checkpoint, model=core, config=search_space)
        model.eval()
        model.freeze()
        model.to('cuda')
        
        dm = ECGDataModule('datasets/hhmusedata', batch_size=50, mode=HDF5ECGDataset.Mode.MODE_HH_CLASSIFIER_SIMPLE,
                       num_workers=7, sample_size=1000, train_fraction=0.8, dev_fraction=0.2, test_fraction=0.0)
        
        val_dataloader = dm.val_dataloader()
        for batch in val_dataloader:
            x, y = batch
            x = x.to('cuda')
            y = y.to('cuda')
            
            y_hat = model(x)
            y_hat = y_hat.argmax(dim=1)
            
            print(y)
            print(y_hat)
            print((y == y_hat).sum())
            print(y.shape)
            print(y_hat.shape)
            break
        
    else:
        train_func(search_space, max_epochs, num_samples)