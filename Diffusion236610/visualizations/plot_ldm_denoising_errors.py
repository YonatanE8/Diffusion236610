from Diffusion236610 import LOGS_DIR
from matplotlib import pyplot as plt
from Diffusion236610.models.diffusion_samplers import DDIMSampler

import os
import pickle
import numpy as np


def generate_loss_per_time_trajectory(loss_per_time_step: dict) -> (np.ndarray, np.ndarray, np.ndarray):
    time = np.array(sorted(loss_per_time_step.keys()))
    losses_per_time = [
        np.array(
            [
                batch[1]
                for batch in loss_per_time_step[t]
            ]
        )
        for t in time
    ]
    loss_mean_per_time = [
        np.mean(loss_per_time)
        for loss_per_time in losses_per_time
    ]
    loss_mean_per_time = np.array(loss_mean_per_time)
    loss_std_per_time = [
        np.std(loss_per_time)
        for loss_per_time in losses_per_time
    ]
    loss_std_per_time = np.array(loss_std_per_time)

    return time, loss_mean_per_time, loss_std_per_time


logs_dir = "/mnt/qnap/yonatane/logs/"
model_name = "ECG_Generation_LDM_SinglePatient_0_FixedUNet1D_2023-04-19"
errors_path = os.path.join(logs_dir, model_name, 'denoiser_performance.pkl')
n_steps = 100
sampler = DDIMSampler(
    model=None,
)
time_steps = sampler.time_steps
alphas = sampler.ddim_alpha
alpha = 0.5
bins = 100
if __name__ == "__main__":
    with open(errors_path, 'rb') as f:
        errors = pickle.load(f)
        train_loss_per_time_step = errors['train_loss_per_time_step']
        test_loss_per_time_step = errors['test_loss_per_time_step']

    time, train_loss_per_time_mean, train_loss_per_time_std = generate_loss_per_time_trajectory(
        train_loss_per_time_step
    )
    _, test_loss_per_time_mean, test_loss_per_time_std = generate_loss_per_time_trajectory(
        test_loss_per_time_step
    )

    train_loss_per_batch = [
        np.array(
            [
                batch[1]
                for batch in train_loss_per_time_step[step]
            ]
        )
        for step in train_loss_per_time_step
    ]
    total_train_loss_per_batch = np.concatenate(train_loss_per_batch, axis=0)

    test_loss_per_batch = [
        np.array(
            [
                batch[1]
                for batch in test_loss_per_time_step[step]
            ]
        )
        for step in test_loss_per_time_step
    ]
    total_test_loss_per_batch = np.concatenate(test_loss_per_batch, axis=0)

    fig = plt.figure()
    plt.hist(total_train_loss_per_batch, bins=bins, alpha=alpha, label=r'$\epsilon_{Train}$ MSE')
    plt.hist(total_test_loss_per_batch, bins=bins, alpha=alpha, label=r'$\epsilon_{Test}$ MSE')
    plt.legend(loc='upper right')
    fig.savefig(
        os.path.join(LOGS_DIR, 'LDM_Train_Test_Denoiser_Errors_Histogram.png'),
        dpi=300,
        format='png',
    )

    fig = plt.figure()
    plt.plot(
        time,
        train_loss_per_time_mean,
        color='b',
        linestyle=':',
        marker='o',
        label=r'$\epsilon_{Train}$ MSE'
    )
    plt.plot(
        time,
        test_loss_per_time_mean,
        color='r',
        linestyle='--',
        marker='x',
        label=r'$\epsilon_{Test}$ MSE'
    )
    plt.legend(loc='upper right')
    fig.savefig(
        os.path.join(LOGS_DIR, 'LDM_Train_Test_Denoiser_Errors_Plot_Per_Time.png'),
        dpi=300,
        format='png',
    )

    plt.show()
