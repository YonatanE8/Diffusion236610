import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

from datetime import datetime
from Diffusion236610 import LOGS_DIR
from torch.utils.data import ConcatDataset
from Diffusion236610.utils.loggers import Logger
from Diffusion236610.utils.optim import Optimizer
from Diffusion236610.utils.trainers import BaseTrainer
from Diffusion236610.utils.schedulers import CycleScheduler
from Diffusion236610.data.datasets import ECGGenerationDataset
from Diffusion236610.utils.utils import get_train_val_test_split
from Diffusion236610.losses.losses import ModuleLoss, VQVAE_Loss
from Diffusion236610.models.ae import Autoencoder, Encoder, Decoder

import torch
import numpy as np

if __name__ == '__main__':
    # Set seed
    seed = 8783
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Define the log dir
    date = str(datetime.today()).split()[0]
    experiment_name = f"ECG_LDM_PRETRAINED_AE_{date}"
    logs_dir = os.path.join(LOGS_DIR, experiment_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Define the training, validation and test sets
    train_set, val_set, test_set = get_train_val_test_split()

    # Define the Datasets & Data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_workers = 32
    prediction_horizon = 0
    window_size = 128 * 1
    samples_overlap = 0
    sample_length = window_size * 1
    n_dims = 3
    train_ratio = 0.95
    val_ratio = 0.025
    keep_buffer = False
    train_datasets = [
        ECGGenerationDataset(
            record_path=record,
            mode='Train',
            prediction_horizon=prediction_horizon,
            samples_overlap=samples_overlap,
            sample_length=sample_length,
            window_size=window_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            n_dims=n_dims,
            keep_buffer=keep_buffer,
        )
        for record in train_set
    ]
    train_ds = ConcatDataset(train_datasets)
    val_datasets = [
        ECGGenerationDataset(
            record_path=record,
            mode='Val',
            prediction_horizon=prediction_horizon,
            samples_overlap=samples_overlap,
            sample_length=sample_length,
            window_size=window_size,
            train_ratio=val_ratio,
            val_ratio=train_ratio,
            n_dims=n_dims,
            keep_buffer=keep_buffer,
        )
        for record in val_set
    ]
    val_ds = ConcatDataset(val_datasets)
    pin_memory = True
    drop_last = False
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Define the model
    channels = 32
    channel_multipliers = [1, 1, 1]
    n_resnet_blocks = 1
    in_channels = 2
    z_channels = 32
    replace_strides_with_dilations = False
    encoder = Encoder(
        channels=channels,
        channel_multipliers=channel_multipliers,
        n_resnet_blocks=n_resnet_blocks,
        in_channels=in_channels,
        z_channels=z_channels,
        replace_strides_with_dilations=replace_strides_with_dilations,
    )
    decoder = Decoder(
        channels=(channels * channel_multipliers[-1]),
        channel_multipliers=channel_multipliers[::-1],
        n_resnet_blocks=n_resnet_blocks,
        out_channels=in_channels,
        z_channels=z_channels,
        replace_strides_with_dilations=replace_strides_with_dilations,
    )
    model_params = {
        'channels': channels,
        'channel_multipliers': channel_multipliers,
        'n_resnet_blocks': n_resnet_blocks,
        'in_channels': in_channels,
        'z_channels': z_channels,
        'replace_strides_with_dilations': replace_strides_with_dilations,
    }

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        z_channels=z_channels,
    )
    model.to(device)

    # Define the optimizer
    lr = 0.001
    weight_decay = 0.01
    optimizer_hparams = {
        'lr': lr,
        'weight_decay': weight_decay,
    }
    optimizers = [
        torch.optim.AdamW(
            params=model.parameters(),
            **optimizer_hparams,
        ),
    ]
    num_epochs = 25
    warmup_proportion = 0.05
    momentum = None
    ag_scheduler_hparams = {
        'lr_max': lr,
        'n_iter': num_epochs * len(train_ds),
        'warmup_proportion': warmup_proportion,
        'momentum': momentum,
    }
    ag_schedulers = [
        CycleScheduler(
            optimizer=optimizers[-1],
            **ag_scheduler_hparams
        ),
    ]
    optimizer = Optimizer(optimizers=optimizers, agnostic_schedulers=ag_schedulers)
    loss_fn = VQVAE_Loss()
    evaluation_metric = ModuleLoss(
        model=torch.nn.MSELoss(),
        scale=1,
    )

    # Define the logger
    logger = Logger(
        log_dir=LOGS_DIR,
        experiment_name=experiment_name,
        max_elements=2,
    )

    # Define the trainer
    checkpoints = True
    early_stopping = None
    checkpoints_mode = 'min'
    trainer = BaseTrainer(
        model=model,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )

    # Write Scenario Specs
    specs = {
        'Data Specs': '',
        "seed": seed,
        'prediction_horizon': prediction_horizon,
        'samples_overlap': samples_overlap,
        'sample_length': sample_length,
        'window_size': window_size,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'DataLoader Specs': '',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'Model Specs': '',
        'Model': type(model).__name__,
    }
    specs.update(model_params)
    loss_params = {
        'Loss Specs': '',
        'loss_fn': f"{loss_fn}",
        'eval_fn': f"{evaluation_metric}",
        'Trainer Specs': '',
        'num_epochs': num_epochs,
        'checkpoints': checkpoints,
        'early_stopping': early_stopping,
        'checkpoints_mode': checkpoints_mode,
        'Optimizer Specs': '',
        'optimizer': type(optimizers[0]).__name__,
    }
    specs.update(loss_params)
    specs.update(optimizer_hparams)
    specs['LR Scheduler Specs'] = ''
    specs['lr_scheduler'] = type(ag_schedulers[0]).__name__
    specs.update(ag_scheduler_hparams)

    specs_file = os.path.join(logs_dir, 'data_specs.txt')
    with open(specs_file, 'w') as f:
        for k, v in specs.items():
            f.write(f"{k}: {str(v)}\n")

    print("Fitting the model")
    trainer.fit(
        dl_train=train_dl,
        dl_val=val_dl,
        num_epochs=num_epochs,
        checkpoints=checkpoints,
        checkpoints_mode=checkpoints_mode,
        early_stopping=early_stopping,
    )

    # Define the test-set
    print("Evaluating over the test set")
    test_datasets = [
        ECGGenerationDataset(
            record_path=record,
            mode='Test',
            prediction_horizon=prediction_horizon,
            samples_overlap=samples_overlap,
            sample_length=sample_length,
            window_size=window_size,
            train_ratio=val_ratio,
            val_ratio=val_ratio,
            n_dims=n_dims,
            keep_buffer=keep_buffer,
        )
        for record in test_set
    ]
    test_ds = ConcatDataset(test_datasets)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        z_channels=z_channels,
    )
    model_ckpt_path = f"{logs_dir}/BestModel.PyTorchModule"  # loading best model
    model_ckp = torch.load(model_ckpt_path)
    model.load_state_dict(model_ckp['model'])
    model.to(device)
    trainer = BaseTrainer(
        model=model,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )

    # Evaluate
    trainer.evaluate(
        dl_test=test_dl,
        ignore_cap=True,
    )
