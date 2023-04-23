# Implementation is heavily based on https://github.com/labmlai/annotated_deep_learning_paper_implementations
# with minor edits to make it a 1D model
from typing import Optional, List
from Diffusion236610.models.ldm import LatentDiffusionModel
from Diffusion236610.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY

import torch
import numpy as np


class DiffusionSampler:
    """
    Base class for sampling algorithms
    """

    def __init__(
            self,
            model: LatentDiffusionModel,
            n_steps: int = 1000,
            linear_start: float = 0.001,
            linear_end: float = 0.05,
    ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """

        super().__init__()

        # Set the model $\epsilon_\text{cond}(x_t, c)$
        self.model = model

        # Get number of steps the model was trained with $T$
        self.n_steps = n_steps
        self.linear_start = linear_start
        self.linear_end = linear_end

        # Set time bar
        self.time_steps = np.linspace(start=0, stop=1, num=n_steps)

        # $\beta$ schedule
        beta = torch.linspace(
            linear_start ** 0.5,
            linear_end ** 0.5,
            n_steps,
            dtype=torch.float32,
            requires_grad=False,
        ) ** 2
        self.beta = beta

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - beta

        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar = alpha_bar

    def get_eps(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            c: torch.Tensor,
            *,
            uncond_scale: float,
            uncond_cond: Optional[torch.Tensor],
    ):
        """
        Get $\epsilon(x_t, c)$
        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param t: is $t$ of shape `[batch_size]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        # When the scale $s = 1$
        # $$\epsilon_\theta(x_t, c) = \epsilon_\text{cond}(x_t, c)$$
        if uncond_cond is None or uncond_scale == 1.:
            model_out = self.model(x, t, c)
            return model_out[MODELS_TENSOR_PREDICITONS_KEY]

        # Duplicate $x_t$ and $t$
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)

        # Concatenated $c$ and $c_u$
        c_in = torch.cat([uncond_cond, c])

        # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
        eps = self.model(x_in, t_in, c_in)[MODELS_TENSOR_PREDICITONS_KEY]

        # Calculate
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        e_t = uncond_scale * eps

        return e_t

    def __call__(
            self,
            shape: List[int],
            cond: torch.Tensor,
            repeat_noise: bool = False,
            temperature: float = 1.,
            x_last: Optional[torch.Tensor] = None,
            uncond_scale: float = 1.,
            uncond_cond: Optional[torch.Tensor] = None,
            skip_steps: int = 0,
    ):
        return self.sample(
            shape=shape,
            cond=cond,
            repeat_noise=repeat_noise,
            temperature=temperature,
            x_last=x_last,
            uncond_scale=uncond_scale,
            uncond_cond=uncond_cond,
            skip_steps=skip_steps,
        )

    def sample(
            self,
            shape: List[int],
            cond: torch.Tensor,
            repeat_noise: bool = False,
            temperature: float = 1.,
            x_last: Optional[torch.Tensor] = None,
            uncond_scale: float = 1.,
            uncond_cond: Optional[torch.Tensor] = None,
            skip_steps: int = 0,
    ):
        """
        ### Sampling Loop
        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param repeat_noise: whether to use the same noise for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip.
        """

        raise NotImplementedError()

    def q_sample(
            self,
            x0: torch.Tensor,
            index: int,
            noise: Optional[torch.Tensor] = None,
    ):
        """
        Sample from $q(x_t|x_0)$
        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """

        raise NotImplementedError()


class DDIMSampler(DiffusionSampler):
    """
    DDIM Sampler
    This extends the `DiffusionSampler` base class.
    DDPM samples images by repeatedly removing noise by sampling step by step using,
    \begin{align}
    x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
            \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
            \Bigg) \\
            &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
            &+ \sigma_{\tau_i} \epsilon_{\tau_i}
    \end{align}
    where $\epsilon_{\tau_i}$ is random noise,
    $\tau$ is a subsequence of $[1,2,\dots,T]$ of length $S$,
    and
    $$\sigma_{\tau_i} =
    \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
    \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$
    Note that, $\alpha_t$ in DDIM paper refers to ${\color{lightgreen}\bar\alpha_t}$ from [DDPM](ddpm.html).
    """

    def __init__(
            self,
            model: LatentDiffusionModel,
            n_steps: int = 100,
            linear_start: float = 0.001,
            linear_end: float = 0.05,
            ddim_discretize: str = "uniform",
            ddim_eta: float = 0.1,
    ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        :param n_steps: is the number of DDIM sampling steps, $S$
        :param ddim_discretize: specifies how to extract $\tau$ from $[1,2,\dots,T]$.
            It can be either `uniform` or `quad`.
        :param ddim_eta: is $\eta$ used to calculate $\sigma_{\tau_i}$. $\eta = 0$ makes the
            sampling process deterministic.
        """

        super().__init__(
            model=model,
            n_steps=n_steps,
            linear_start=linear_start,
            linear_end=linear_end,
        )

        self._ddim_discretize = ddim_discretize

        # Calculate $\tau$ to be uniformly distributed across $[1,2,\dots,T]$
        if ddim_discretize == 'uniform':
            self.time_steps = np.asarray(list(range(0, self.n_steps)))

        # Calculate $\tau$ to be quadratically distributed across $[1,2,\dots,T]$
        elif ddim_discretize == 'quad':
            self.time_steps = ((np.linspace(0, np.sqrt(self.n_steps * .8))) ** 2).astype(int)

        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            # $\alpha_{\tau_i}$
            self.ddim_alpha = self.alpha_bar[self.time_steps].clone().to(torch.float32)

            # $\sqrt{\alpha_{\tau_i}}$
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)

            # $\alpha_{\tau_{i-1}}$
            self.ddim_alpha_prev = torch.cat([self.alpha_bar[0:1], self.alpha_bar[self.time_steps[:-1]]])

            # $$\sigma_{\tau_i} =
            # \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}}
            # \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$$
            self.ddim_sigma = (
                    ddim_eta *
                    (
                            (1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                            (1 - self.ddim_alpha / self.ddim_alpha_prev)
                    ) ** .5
            )

            # $\sqrt{1 - \alpha_{\tau_i}}$
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** .5

    @torch.no_grad()
    def sample(
            self,
            shape: List[int],
            cond: torch.Tensor,
            repeat_noise: bool = False,
            temperature: float = 1.,
            x_last: Optional[torch.Tensor] = None,
            uncond_scale: float = 1.,
            uncond_cond: Optional[torch.Tensor] = None,
            skip_steps: int = 0,
            time_steps: Optional[int] = None,
    ):
        """
        Sampling Loop
        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, length]`
        :param cond: is the conditional embeddings $c$
        :param repeat_noise: whether to use the same noise for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_{\tau_S}$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $i'$. We start sampling from $S - i'$.
            And `x_last` is then $x_{\tau_{S - i'}}$.
        :param time_steps: Controls the number of time steps for the current specific sample
        """

        # Set the appropriate time steps
        if time_steps is not None:
            # Calculate $\tau$ to be uniformly distributed across $[1,2,\dots,T]$
            if self._ddim_discretize == 'uniform':
                time_steps = np.asarray(list(range(0, time_steps)))

            # Calculate $\tau$ to be quadratically distributed across $[1,2,\dots,T]$
            elif self._ddim_discretize == 'quad':
                time_steps = ((np.linspace(0, np.sqrt(time_steps * .8))) ** 2).astype(int)

        else:
            time_steps = self.time_steps

        # Get device and batch size
        device = self.model.device
        bs = shape[0]

        # Get $x_{\tau_S}$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at $\tau_{S - i'}, \tau_{S - i' - 1}, \dots, \tau_1$
        time_steps = np.flip(time_steps)[skip_steps:]

        for i, step in enumerate(time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            index = len(time_steps) - i - 1

            # Time step $\tau_i$
            ts = x.new_full(
                (bs,),
                step,
                dtype=torch.long,
            )

            # Sample $x_{\tau_{i-1}}$
            x, pred_x0, e_t = self.p_sample(
                x=x,
                c=cond,
                t=ts,
                step=step,
                index=index,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )

        # Return $x_0$
        return x

    @torch.no_grad()
    def p_sample(
            self,
            x: torch.Tensor,
            c: torch.Tensor,
            t: torch.Tensor,
            step: int,
            index: int,
            *,
            repeat_noise: bool = False,
            temperature: float = 1.,
            uncond_scale: float = 1.,
            uncond_cond: Optional[torch.Tensor] = None,
    ):
        """
        Sample $x_{\tau_{i-1}}$
        :param x: is $x_{\tau_i}$ of shape `[batch_size, channels, length`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $\tau_i$ of shape `[batch_size]`
        :param step: is the step $\tau_i$ as an integer
        :param index: is index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
        :param repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        # Get $\epsilon_\theta(x_{\tau_i}}$
        e_t = self.get_eps(
            x=x,
            t=t,
            c=c,
            uncond_scale=uncond_scale,
            uncond_cond=uncond_cond,
        )

        # Calculate $x_{\tau_{i - 1}}$ and predicted $x_0$
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(
            e_t=e_t,
            index=index,
            x=x,
            temperature=temperature,
            repeat_noise=repeat_noise,
        )

        return x_prev, pred_x0, e_t

    def get_x_prev_and_pred_x0(
            self,
            e_t: torch.Tensor,
            index: int,
            x: torch.Tensor,
            *,
            temperature: float,
            repeat_noise: bool,
    ):
        """
        $x_{\tau_{i-1}}$ given $\epsilon_\theta(x_{\tau_i}}$
        """

        # $\alpha_{\tau_i}$
        alpha = self.ddim_alpha[index]

        # $\alpha_{\tau_{i-1}}$
        alpha_prev = self.ddim_alpha_prev[index]

        # $\sigma_{\tau_i}$
        sigma = self.ddim_sigma[index]

        # $\sqrt{1 - \alpha_{\tau_i}}$
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]

        # Current prediction for $x_0$,
        # $$\frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}$$
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)

        # Direction pointing to $x_t$
        # $$\sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i})$$
        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        # No noise is added, when $\eta = 0$
        if sigma == 0.:
            noise = 0.

        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)

        # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=x.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        #  \begin{align}
        #     x_{\tau_{i-1}} &= \sqrt{\alpha_{\tau_{i-1}}}\Bigg(
        #             \frac{x_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}}\epsilon_\theta(x_{\tau_i})}{\sqrt{\alpha_{\tau_i}}}
        #             \Bigg) \\
        #             &+ \sqrt{1 - \alpha_{\tau_{i- 1}} - \sigma_{\tau_i}^2} \cdot \epsilon_\theta(x_{\tau_i}) \\
        #             &+ \sigma_{\tau_i} \epsilon_{\tau_i}
        #  \end{align}
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev, pred_x0

    @torch.no_grad()
    def q_sample(
            self,
            x0: torch.Tensor,
            index: int,
            noise: Optional[torch.Tensor] = None,
    ):
        """
        Sample from $q_{\sigma,\tau}(x_{\tau_i}|x_0)$
        $$q_{\sigma,\tau}(x_t|x_0) =
         \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$
        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $\tau_i$ index $i$
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from
        #  $$q_{\sigma,\tau}(x_t|x_0) =
        #          \mathcal{N} \Big(x_t; \sqrt{\alpha_{\tau_i}} x_0, (1-\alpha_{\tau_i}) \mathbf{I} \Big)$$
        return self.ddim_alpha_sqrt[index] * x0 + self.ddim_sqrt_one_minus_alpha[index] * noise


class DDPMSampler(DiffusionSampler):
    """
    DDPM Sampler
    This extends the [`DiffusionSampler` base class](index.html).
    DDPM samples images by repeatedly removing noise by sampling step by step from
    $p_\theta(x_{t-1} | x_t)$,
    \begin{align}
    p_\theta(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big) \\
    \mu_t(x_t, t) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
                         + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\
    \tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t \\
    x_0 &= \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta \\
    \end{align}
    """

    def __init__(
            self,
            model: LatentDiffusionModel,
            n_steps: int = 1000,
            linear_start: float = 0.001,
            linear_end: float = 0.05,
    ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """

        super().__init__(
            model=model,
            n_steps=n_steps,
            linear_start=linear_start,
            linear_end=linear_end,
        )

        with torch.no_grad():
            # $\beta_t$ schedule
            beta = self.model.beta

            # $\bar\alpha_{t-1}$
            alpha_bar_prev = torch.cat([self.alpha_bar.new_tensor([1.]), self.alpha_bar[:-1]])

            # $\sqrt{\bar\alpha}$
            self.sqrt_alpha_bar = self.alpha_bar ** .5

            # $\sqrt{1 - \bar\alpha}$
            self.sqrt_1m_alpha_bar = (1. - self.alpha_bar) ** .5

            # $\frac{1}{\sqrt{\bar\alpha_t}}$
            self.sqrt_recip_alpha_bar = self.alpha_bar ** -.5

            # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
            self.sqrt_recip_m1_alpha_bar = (1 / self.alpha_bar - 1) ** .5

            # $\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$
            variance = beta * (1. - alpha_bar_prev) / (1. - self.alpha_bar)

            # Clamped log of $\tilde\beta_t$
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))

            # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
            self.mean_x0_coef = beta * (alpha_bar_prev ** .5) / (1. - self.alpha_bar)

            # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
            self.mean_xt_coef = (1. - alpha_bar_prev) * ((1 - beta) ** 0.5) / (1. - self.alpha_bar)

    @torch.no_grad()
    def sample(
            self,
            shape: List[int],
            cond: torch.Tensor,
            repeat_noise: bool = False,
            temperature: float = 1.,
            x_last: Optional[torch.Tensor] = None,
            uncond_scale: float = 1.,
            uncond_cond: Optional[torch.Tensor] = None,
            skip_steps: int = 0,
    ):
        """
        Sampling Loop
        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param repeat_noise: whether to use the same noise for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $t'$. We start sampling from $T - t'$.
            And `x_last` is then $x_{T - t'}$.
        """

        # Get device and batch size
        device = self.model.device
        bs = shape[0]

        # Get $x_T$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        # Sampling loop
        for step in time_steps:
            # Time step $t$
            ts = x.new_full(
                (bs,),
                step,
                dtype=torch.long,
            )

            # Sample $x_{t-1}$
            x, pred_x0, e_t = self.p_sample(
                x=x,
                c=cond,
                t=ts,
                step=step,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )

        # Return $x_0$
        return x

    @torch.no_grad()
    def p_sample(
            self,
            x: torch.Tensor,
            c: torch.Tensor,
            t: torch.Tensor,
            step: int,
            repeat_noise: bool = False,
            temperature: float = 1.,
            uncond_scale: float = 1.,
            uncond_cond: Optional[torch.Tensor] = None,
    ):
        """
        ### Sample $x_{t-1}$ from $p_\theta(x_{t-1} | x_t)$
        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $t$ of shape `[batch_size]`
        :param step: is the step $t$ as an integer
        :param repeat_noise: whether to use the same noise for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        # Get $\epsilon_\theta$
        e_t = self.get_eps(
            x=x,
            t=t,
            c=c,
            uncond_scale=uncond_scale,
            uncond_cond=uncond_cond,
        )

        # Get batch size
        bs = x.shape[0]

        # $\frac{1}{\sqrt{\bar\alpha_t}}$
        sqrt_recip_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step])

        # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
        sqrt_recip_m1_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step])

        # Calculate $x_0$ with current $\epsilon_\theta$
        # $$x_0 = \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta$$
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t

        # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])

        # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])

        # Calculate $\mu_t(x_t, t)$
        # $$\mu_t(x_t, t) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #    + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$
        mean = mean_x0_coef * x0 + mean_xt_coef * x

        # $\log \tilde\beta_t$
        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])

        # Do not add noise when $t = 1$ (final step sampling process).
        # Note that `step` is `0` when $t = 1$)
        if step == 0:
            noise = 0

        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]))

        # Different noise for each sample
        else:
            noise = torch.randn(x.shape)

        # Multiply noise by the temperature
        noise = noise * temperature

        # Sample from,
        # $$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big)$$
        x_prev = mean + (0.5 * log_var).exp() * noise

        return x_prev, x0, e_t

    @torch.no_grad()
    def q_sample(
            self,
            x0: torch.Tensor,
            index: int,
            noise: Optional[torch.Tensor] = None,
    ):
        """
        Sample from $q(x_t|x_0)$
        $$q(x_t|x_0) = \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$$
        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from $\mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise
