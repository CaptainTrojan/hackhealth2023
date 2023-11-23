from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune

from model import CDIL, AutoEncoder
from dataloader import ECGDataModule
import argparse
import ray
from ray.tune.schedulers import ASHAScheduler
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, CheckpointConfig, ScalingConfig


def train_func(config):
    dm = ECGDataModule('datasets/ikem', batch_size=config['batch_size'], mode=HDF5ECGDataset.Mode.MODE_MASKED_AUTOENCODER)
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
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)
    
    return trainer.checkpoint_callback.best_model_path


if __name__ == '__main__':    
    # setup argparse to load ray address and smoke test boolean flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    if args.ray_address:
        ray.init(address=args.ray_address, local_mode=True)
    else:
        ray.init(local_mode=True, log_to_driver=False)
    
    if args.smoke_test:
        max_epochs = 1
        num_samples = 1
    else:
        max_epochs = 100
        num_samples = 100
    
    search_space = {
        "hidden_channels": tune.choice([16, 32, 64, 128]),
        "num_layers": tune.choice([3, 4, 5, 6]),
        "kernel_size": tune.choice([3, 5, 7, 9]),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "learning_rate": tune.loguniform(5e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
    }
    scheduler = ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2)
    
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 10, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
        storage_path="/tmp/ray_tmp/ray_results",
    )
    
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    tuner = tune.Tuner(
        ray_trainer,
        param_space={'train_loop_config': search_space},
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_loss",
            mode="min",
            scheduler=scheduler
        ),
    )
    
    results = tuner.fit()
    best_trial = results.get_best_result("val_loss", "min", "last")
    print("Best trial {}".format(best_trial))