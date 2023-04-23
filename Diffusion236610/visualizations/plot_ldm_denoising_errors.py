from Diffusion236610 import LOGS_DIR
from matplotlib import pyplot as plt
from Diffusion236610.models.diffusion_samplers import DDIMSampler

import os
import pickle
import numpy as np

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
        os.path.join(LOGS_DIR, 'LDM_Train_Test_Denoiser_Errors.png'),
        dpi=300,
        format='png',
    )
    plt.show()

