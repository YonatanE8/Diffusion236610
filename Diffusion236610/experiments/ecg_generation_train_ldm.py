import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

from datetime import datetime
from Diffusion236610 import LOGS_DIR
from torch.utils.data import ConcatDataset
from Diffusion236610.utils.loggers import Logger
from Diffusion236610.utils.optim import Optimizer
from Diffusion236610.utils.trainers import LDMTrainer
from Diffusion236610.utils.schedulers import CycleScheduler
from Diffusion236610.losses.losses import ModuleLoss, LDMLoss
from Diffusion236610.models.ldm import LatentDiffusionModel
from Diffusion236610.utils.utils import get_train_val_test_split
from Diffusion236610.models.diffusion_samplers import DDIMSampler
from Diffusion236610.models.ae import Autoencoder, Encoder, Decoder
from Diffusion236610.data.datasets import ECGGenerationDataset
from Diffusion236610.models.unet import UNet1D, SelfAttentionConditioningModule, FixedUNet1D

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
    fit_model = False
    single_patient = True
    single_patient_id = 0
    fixed_unet = True
    scale_latent_space = False
    experiment_name = f"ECG_Generation_LDM_" \
                      f"{f'SinglePatient_{single_patient_id}_' if single_patient else ''}" \
                      f"{'FixedUNet1D_' if fixed_unet else ''}" \
                      f"{'NotScaled_' if not scale_latent_space else ''}" \
                      f"{date}"
    logs_dir = os.path.join(LOGS_DIR, experiment_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Define the training, validation and test sets
    train_set, val_set, test_set = get_train_val_test_split()
    if single_patient:
        train_set = [train_set[single_patient_id], ]
        val_set = train_set
        test_set = train_set

    # Define the Datasets & Data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_workers = 32
    prediction_horizon = 1
    window_size = 128 * 5
    samples_overlap = 0
    sample_length = window_size * 1
    n_dims = 3
    if single_patient:
        train_ratio = 0.6
        val_ratio = 0.2

    else:
        train_ratio = 0.95
        val_ratio = 0.025

    leads_as_channels = True
    n_leads = 2
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
            leads_as_channels=leads_as_channels,
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
            train_ratio=val_ratio if not single_patient else train_ratio,
            val_ratio=train_ratio if not single_patient else val_ratio,
            n_dims=n_dims,
            leads_as_channels=leads_as_channels,
            keep_buffer=keep_buffer,
        )
        for record in val_set
    ]
    val_ds = ConcatDataset(val_datasets)
    test_datasets = [
        ECGGenerationDataset(
            record_path=record,
            mode='Test',
            prediction_horizon=prediction_horizon,
            samples_overlap=samples_overlap,
            sample_length=sample_length,
            window_size=window_size,
            train_ratio=val_ratio if not single_patient else train_ratio,
            val_ratio=val_ratio,
            n_dims=n_dims,
            leads_as_channels=leads_as_channels,
            keep_buffer=keep_buffer,
        )
        for record in test_set
    ]
    test_ds = ConcatDataset(test_datasets)

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

    # Define the AE model
    channels = 32
    channel_multipliers = [1, 1, 1]
    n_resnet_blocks = 1
    in_channels = 2
    z_channels = 64
    replace_strides_with_dilations = True
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
    ae_params = {
        'channels': channels,
        'channel_multipliers': channel_multipliers,
        'n_resnet_blocks': n_resnet_blocks,
        'in_channels': in_channels,
        'z_channels': z_channels,
        'replace_strides_with_dilations': replace_strides_with_dilations,
    }

    ae = Autoencoder(
        encoder=encoder,
        decoder=decoder,
        z_channels=z_channels,
    )

    # Load the pre-trained AE parameters
    trained_ae_checkpoint_path = r"/mnt/qnap/yonatane/logs/ldm/ECG_LDM_PRETRAINED_AE_2023-04-10/BestModel.PyTorchModule"
    trained_ae_ckpt = torch.load(trained_ae_checkpoint_path)['model']
    ae.load_state_dict(trained_ae_ckpt)

    # Set requires_grad=False, to not keep training the AE
    ae.requires_grad_(False)

    # Compute the scaling factor
    dummy_input = torch.zeros((1, n_leads, window_size))
    latent_scaling_factor = ae.get_scaling_factor(x=dummy_input)

    # Set the denoising model
    in_channels = z_channels
    out_channels = in_channels
    channels = 32
    n_res_blocks = 2
    channel_multipliers = [1, 1, 1, 1, 1]
    n_heads = 8
    attention_levels = [0, 2, 4]
    tf_layers = 2
    d_cond = z_channels
    use_dilation = False
    bias = False
    dropout = 0.0

    if fixed_unet:
        unet_params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'channels': channels,
            'channels_multiplication_factor': 4,
            'n_transformers_layers': 3,
            'n_heads': n_heads,
            'd_cond': d_cond,
            'use_dilation': use_dilation,
            'bias': bias,
            'dropout': dropout,
        }
        unet_model = FixedUNet1D(
            **unet_params
        )

    else:
        unet_params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'channels': channels,
            'n_res_blocks': n_res_blocks,
            'channel_multipliers': channel_multipliers,
            'n_heads': n_heads,
            'attention_levels': attention_levels,
            'tf_layers': tf_layers,
            'd_cond': d_cond,
            'use_dilation': use_dilation,
            'bias': bias,
            'dropout': dropout,

        }
        unet_model = UNet1D(
            **unet_params
        )

    # Set the conditioning model
    cond_activation = 'leakyrelu'
    cond_params = {
        'input_dim': int(sample_length // latent_scaling_factor),
        'embed_dim': d_cond,
        'num_heads': n_heads,
        'dropout': dropout,
        'bias': bias,
        'activation': cond_activation,
    }
    conditioning_module = SelfAttentionConditioningModule(
        **cond_params
    )

    # Finally, define the entire LDM model
    model = LatentDiffusionModel(
        unet_model=unet_model,
        autoencoder=ae,
        latent_scaling_factor=latent_scaling_factor,
        conditioning_module=conditioning_module,
        device=device,
    )
    model.to(device)

    # Define the diffusion sampler
    n_steps = 100
    linear_start = 0.001
    linear_end = 0.05
    ddim_discretize = "uniform"
    ddim_eta = 0.1
    sampler_params = {
        'n_steps': n_steps,
        'linear_start': linear_start,
        'linear_end': linear_end,
        'ddim_discretize': ddim_discretize,
        'ddim_eta': ddim_eta,
    }
    sampler = DDIMSampler(
        model=model,
        **sampler_params
    )

    # Define the optimizer
    lr = 4.5e-5
    weight_decay = 0.0001
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
    num_epochs = 100
    warmup_proportion = 0.2
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
    loss_fn = LDMLoss(
        scale=1000,
    )
    evaluation_metric = ModuleLoss(
        model=torch.nn.L1Loss(),
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
    max_iterations_per_epoch_test = 10
    trainer = LDMTrainer(
        model=model,
        diffusion_sampler=sampler,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
        max_iterations_per_epoch_test=max_iterations_per_epoch_test
    )

    if fit_model:
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
        specs.update({'AE Specs': ''})
        specs.update(ae_params)
        specs.update({'Denoiser Specs': ''})
        specs.update(unet_params)
        specs.update({'Conditioning Module Specs': ''})
        specs.update(cond_params)
        specs.update({'Sampler Specs': ''})
        specs.update(sampler_params)
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
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    if fixed_unet:
        model = LatentDiffusionModel(
            unet_model=FixedUNet1D(
                **unet_params
            ),
            autoencoder=ae,
            latent_scaling_factor=latent_scaling_factor,
            conditioning_module=SelfAttentionConditioningModule(
                **cond_params
            ),
            scale_latent_space=scale_latent_space,
            device=device,
        )

    else:
        model = LatentDiffusionModel(
            unet_model=UNet1D(
                **unet_params
            ),
            autoencoder=ae,
            latent_scaling_factor=latent_scaling_factor,
            conditioning_module=SelfAttentionConditioningModule(
                **cond_params
            ),
            scale_latent_space=scale_latent_space,
            device=device,
        )

    model_ckpt_path = f"{logs_dir}/BestModel.PyTorchModule"
    model_ckp = torch.load(model_ckpt_path)
    model.load_state_dict(model_ckp['model'])
    model.to(device)
    max_iterations_per_epoch_eval = 500
    trainer = LDMTrainer(
        model=model,
        diffusion_sampler=sampler,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
        max_iterations_per_epoch_eval=max_iterations_per_epoch_eval,
    )

    # Evaluate
    trainer.evaluate(
        dl_test=test_dl,
        ignore_cap=True,
    )
