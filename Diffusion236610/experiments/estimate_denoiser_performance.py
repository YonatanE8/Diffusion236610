import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

from torch.utils.data import ConcatDataset
from Diffusion236610.models.ldm import LatentDiffusionModel
from Diffusion236610.data.datasets import ECGGenerationDataset
from Diffusion236610.utils.utils import get_train_val_test_split
from Diffusion236610.models.diffusion_samplers import DDIMSampler
from Diffusion236610.models.ae import Autoencoder, Encoder, Decoder
from Diffusion236610.models.unet import UNet1D, SelfAttentionConditioningModule, FixedUNet1D
from Diffusion236610.utils.defaults import (
    MODELS_TENSOR_PREDICITONS_KEY,
    GT_TENSOR_INPUTS_KEY,
    GT_TENSOR_PREDICITONS_KEY,
)

import torch
import pickle
import numpy as np


if __name__ == '__main__':
    # Set seed
    seed = 8783
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Define the log dir
    date = "2023-04-19"
    single_patient = True
    single_patient_id = 0
    fixed_unet = True
    experiment_name = f"ECG_Generation_LDM_" \
                      f"{f'SinglePatient_{single_patient_id}_' if single_patient else ''}" \
                      f"{'FixedUNet1D_' if fixed_unet else ''}" \
                      f"{date}"
    LOGS_DIR = '/mnt/qnap/yonatane/logs/'
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
        shuffle=False,
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
    test_dl = torch.utils.data.DataLoader(
        test_ds,
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
    # use_dilation = True
    # bias = True
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

    # Load the pre-trained model
    checkpoint_path = os.path.join(logs_dir, "BestModel.PyTorchModule")
    ckpt = torch.load(checkpoint_path)['model']
    model.load_state_dict(ckpt)

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
    evaluation_metric = torch.nn.MSELoss()

    train_loss_per_time_step = {}
    test_loss_per_time_step = {}
    train_dl_iter = iter(train_dl)
    n_train_batches = len(train_dl_iter)
    test_dl_iter = iter(test_dl)
    n_test_batches = len(test_dl_iter)
    with torch.no_grad():
        for b in range(n_train_batches):
            # Unpack the batch
            batch = next(train_dl_iter)
            cond = batch[GT_TENSOR_INPUTS_KEY].to(device)
            x = batch[GT_TENSOR_PREDICITONS_KEY].to(device)

            # Encode the signal
            z = model.autoencoder_encode(x)
            encoded_cond = model.autoencoder_encode(cond)

            # Run the sampler from t=0 to t=T
            for i in range(n_steps):
                print(f"Evaluating t = {i + 1}/{n_steps}, of training batch {b + 1} / {n_train_batches}")
                time_steps = i

                # Generate noise
                noise_scales = sampler.ddim_sigma[time_steps].to(device)
                noise = noise_scales[None, None, None] * torch.randn(z.shape, device=device, dtype=z.dtype)

                # Generate the noisy signal
                noisy_signal = z + noise

                # Predict the noise
                t = (
                        torch.from_numpy(sampler.time_steps[[time_steps]]).to(device) +
                        torch.zeros((z.shape[0],), device=device, dtype=z.dtype)
                )
                predicted_noise = model.forward(
                    x=noisy_signal,
                    t=t,
                    context=encoded_cond,
                )
                predicted_noise = predicted_noise[MODELS_TENSOR_PREDICITONS_KEY]
                loss_scale = (noise_scales ** 2) / (
                        sampler.ddim_alpha[time_steps].to(device) ** 2
                )
                loss = evaluation_metric(
                    predicted_noise,
                    noise,
                )

                if i not in train_loss_per_time_step:
                    train_loss_per_time_step[i] = [(loss_scale.detach().cpu().item(), loss.detach().cpu().item())]

                else:
                    train_loss_per_time_step[i].append((loss_scale.detach().cpu().item(), loss.detach().cpu().item()))

        for b in range(n_test_batches):
            # Unpack the batch
            batch = next(test_dl_iter)
            cond = batch[GT_TENSOR_INPUTS_KEY].to(device)
            x = batch[GT_TENSOR_PREDICITONS_KEY].to(device)

            # Encode the signal
            z = model.autoencoder_encode(x)
            encoded_cond = model.autoencoder_encode(cond)

            # Run the sampler from t=0 to t=T
            for i in range(n_steps):
                print(f"Evaluating t = {i + 1}/{n_steps}, of test batch {b + 1} / {n_test_batches}")
                time_steps = i

                # Generate noise
                noise_scales = sampler.ddim_sigma[time_steps].to(device)
                noise = noise_scales[None, None, None] * torch.randn(z.shape, device=device, dtype=z.dtype)

                # Generate the noisy signal
                noisy_signal = z + noise

                # Predict the noise
                t = (
                        torch.from_numpy(sampler.time_steps[[time_steps]]).to(device) +
                        torch.zeros((z.shape[0],), device=device, dtype=z.dtype)
                )
                predicted_noise = model.forward(
                    x=noisy_signal,
                    t=t,
                    context=encoded_cond,
                )
                predicted_noise = predicted_noise[MODELS_TENSOR_PREDICITONS_KEY]
                loss_scale = (noise_scales ** 2) / (
                        sampler.ddim_alpha[time_steps].to(device) ** 2
                )
                loss = evaluation_metric(
                    predicted_noise,
                    noise,
                )

                if i not in test_loss_per_time_step:
                    test_loss_per_time_step[i] = [(loss_scale.detach().cpu().item(), loss.detach().cpu().item())]

                else:
                    test_loss_per_time_step[i].append((loss_scale.detach().cpu().item(), loss.detach().cpu().item()))

        print("Completed evaluation, saving results...")

        with open(os.path.join(logs_dir, 'denoiser_performance.pkl'), 'wb') as f:
            pickle.dump(
                obj={
                    'train_loss_per_time_step': train_loss_per_time_step,
                    'test_loss_per_time_step': test_loss_per_time_step,
                },
                file=f,
            )


