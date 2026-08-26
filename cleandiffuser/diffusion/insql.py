from typing import Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from cleandiffuser.classifier import BaseClassifier
from cleandiffuser.nn_condition import BaseNNCondition
from cleandiffuser.nn_diffusion import BaseNNDiffusion
from cleandiffuser.utils import (
    at_least_ndim,
    SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE)
from .basic import DiffusionModel
import random


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    # delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    delta_sq = (error.pow(2)).flatten(1).mean(dim=1)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class InsQL(DiffusionModel):
    """Continuous-time Instant Flow Q-Learning (InsQL) policy.

    InsQL parameterizes the policy with the *average velocity* `u(a_t, n, t; s)` over a
    finite interval `[n, t]`, rather than the instantaneous marginal velocity used by
    flow matching. Because the average velocity already integrates the flow across the
    interval, an action is produced by a single query `pi(s) = eps - u(eps, 0, 1; s)`.

    Training uses a compositional objective: the average velocity over a long interval is
    the convex combination of average velocities over two sub-intervals, with
    `beta = (m - n) / (t - n)`. A boundary term anchors the short-interval limit to the
    conditional velocity from flow matching, so no pretrained teacher is required.
    The two terms are mixed by `flow_ratio`, which is `lambda` in the paper.

    This removes solver unrolling and backpropagation-through-time from the actor update,
    and needs no distillation, multi-stage pipeline, or Jacobian computation.

    Args:
    - nn_diffusion: BaseNNDiffusion
        The neural network backbone for the Diffusion model.
    - nn_condition: Optional[BaseNNCondition]
        The neural network backbone for the condition embedding.
        
    - fix_mask: Union[list, np.ndarray, torch.Tensor]
        Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
        The mask should be in the shape of `x_shape`.
    - loss_weight: Union[list, np.ndarray, torch.Tensor]
        Add loss weight. The weight should be in the shape of `x_shape`.
        
    - classifier: Optional[BaseClassifier]
        Add a classifier to enable classifier-guidance.
        
    - grad_clip_norm: Optional[float]
        Gradient clipping norm.
    - ema_rate: float
        Exponential moving average rate.
    - optim_params: Optional[dict]
        Optimizer parameters.
        
    - x_max: Optional[torch.Tensor]
        The maximum value for the input data. `None` indicates no constraint.
    - x_min: Optional[torch.Tensor]
        The minimum value for the input data. `None` indicates no constraint.
        
    - device: Union[torch.device, str]
        The device to run the model.
    """
    def __init__(
            self,

            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ----------------- Masks ----------------- #
            # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
            fix_mask: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`
            # Add loss weight
            loss_weight: Union[list, np.ndarray, torch.Tensor] = None,  # be in the shape of `x_shape`

            # ------------------ Plugins ---------------- #
            # Add a classifier to enable classifier-guidance
            classifier: Optional[BaseClassifier] = None,

            # ------------------ Training Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            # ------------------- Diffusion Params ------------------- #
            x_max: Optional[torch.Tensor] = None,
            x_min: Optional[torch.Tensor] = None,

            device: Union[torch.device, str] = "cpu",

            # ---- InsQL Params ---- #
            adaptive_l2_loss: bool = False,
            uncond_value: float = 0.0,
            time_dist=['lognorm', -0.4, 1.0],
            flow_ratio: float = 0.5, # How much r == t. 
            cfg_w = 1.0,
            cfg_dropout = 0.2, # Dropout ratio for cfg
            warm_up_steps = 100000,
            cfg_uncond='u',
            jvp_api='autograd',
    ):
        super().__init__(
            nn_diffusion, nn_condition, fix_mask, loss_weight, classifier, grad_clip_norm,
            0, ema_rate, optim_params, device)

        assert classifier is None, "InsQL does not support classifier-guidance."

        # InsQL Params
        self.time_dist = time_dist
        self.flow_ratio = flow_ratio
        self.cfg_w = cfg_w
        self.adaptive_l2_loss = adaptive_l2_loss
        self.warm_up_steps = warm_up_steps # warm up by training FM only.
        self.warm_up_count = 0
        # only support cfg_w == 1.0
        assert cfg_w == 1.0, "Split InsQL only supports cfg_w == 1.0 for now."
        # cfg_dropout is the dropout ratio for cfg
        self.cfg_dropout = cfg_dropout
        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api
        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        self.uncond_value = uncond_value

        self.x_max, self.x_min = x_max, x_min

    @property
    def supported_solvers(self):
        return ["euler"]

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    def sample_t_r(self, batch_size, device):
        """Sample t and r from the time distribution. r always less than or equal to t. """

        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # num_selected = int(self.flow_ratio * batch_size)
        # indices = np.random.permutation(batch_size)[:num_selected]
        # r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
    # ==================== Training: Straighten Flow ======================
    
    def add_noise(
        self,
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
    ):
        """Map x0 to xt.

        Args:
            x0 (torch.Tensor): Clean data.
            t (torch.Tensor): Diffusion timestep. Defaults to None.
            eps (torch.Tensor, optional): Noise. Defaults to None.

        Returns:
            xt (torch.Tensor): Noisy data.
            t (torch.Tensor): Diffusion timestep.
            eps (torch.Tensor): Noise.
        """
        batch_size = x0.shape[0]
        device = x0.device
        if t is None or r is None:
            t,r = self.sample_t_r(batch_size, device)

        eps = torch.randn_like(x0) if eps is None else eps

        xt = x0 + at_least_ndim(t, x0.dim()) * (eps - x0)
        xt = xt * (1.0 - self.fix_mask) + x0 * self.fix_mask

        return xt, t, r, eps
    
    def loss(self, x0, x1=None, condition=None):

        # x1 is the samples of source distribution.
        # If x1 is None, then we assume x1 is from a standard Gaussian distribution.
        if x1 is None:
            x1 = torch.randn_like(x0)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        batch_size = x0.shape[0]
        device = x0.device

        normal_fm_obj = self.flow_ratio >0.0 and  random.random() < self.flow_ratio

        if self.warm_up_steps > 0 and self.warm_up_count < self.warm_up_steps:
            normal_fm_obj = True
            self.warm_up_count += 1
            # log warm up count to wandb
            print(f"Warm up count: {self.warm_up_count}/{self.warm_up_steps}")
        
        
        t,r = self.sample_t_r(batch_size, device)

        if normal_fm_obj:
            r = t.clone()
        
        xt = x0 + at_least_ndim(t, x0.dim()) * (x1 - x0)

        xt = xt * (1. - self.fix_mask) + x0 * self.fix_mask

        v = x1 - x0 # velocity is from noise to data.

        if condition is not None:
            assert condition.shape[0] == batch_size, "Condition must have the same batch size as x0"
            condition = self.model["condition"](condition)
            if self.cfg_w != 1.0:
                # TODO: support cfg_w != 1.0
                raise NotImplementedError("InsQL does not support classifier-free guidance for now. Will support in the future.")

            else:
                v_hat = v

        if normal_fm_obj:
            v_pred = self.model["diffusion"](xt, t, r, condition)
            
        else:
            # split mean flow
            lambda_split = torch.rand(batch_size, device=device)
            s_split_time =   (1 - lambda_split) * t + lambda_split *r

            with torch.no_grad():
                u_ts = self.model["diffusion"](xt, t, s_split_time, condition)
            z_s = xt - at_least_ndim(t - s_split_time, x0.dim()) * u_ts
            with torch.no_grad():
                u_sr = self.model["diffusion"](z_s, s_split_time, r, condition)
            v_hat = at_least_ndim((1- lambda_split), x0.dim()) * u_sr + at_least_ndim(lambda_split,x0.dim()) * u_ts
            v_pred = self.model["diffusion"](xt, t, r, condition)
            # compute u(zt)

        error = v_pred - stopgrad(v_hat)
        if self.adaptive_l2_loss:
            loss = adaptive_l2_loss(error, gamma=0.0, c=1e-3)
        else:
            loss = F.mse_loss(v_pred, stopgrad(v_hat), reduction='mean')
        
        is_flow_loss = normal_fm_obj
        return loss, is_flow_loss
        # TODO: support loss_weight and fix_mask
        # return (loss * self.loss_weight * (1 - self.fix_mask)).mean()

    def update(self, x0, condition=None, update_ema=True, x1=None, **kwargs):
        """One-step gradient update.
        Inputs:
        - x0: torch.Tensor
            Samples from the target distribution.
        - condition: Optional
            Condition of x0. `None` indicates no condition.
        - update_ema: bool
            Whether to update the exponential moving average model.
        - x1: torch.Tensor
            Samples from the source distribution. `None` indicates standard Gaussian samples.

        Outputs:
        - log: dict
            The log dictionary.
        """
        loss, is_flow_loss = self.loss(x0, x1, condition)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()

        if update_ema:
            self.ema_update()

        if is_flow_loss:
            log = {"fm_loss": loss.item(), "grad_norm": grad_norm}
        else:
            log = {"loss": loss.item(), "grad_norm": grad_norm}

        return log

    # ==================== Sampling: Solving a straight ODE flow ======================

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            x1: torch.Tensor = None,
            # ----------------- sampling ----------------- #
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
            use_ema: bool = False,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 1.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ----------- Diffusion-X sampling ----------
            diffusion_x_sampling_steps: int = 0,
            # ----------- Warm-Starting -----------
            warm_start_reference: Optional[torch.Tensor] = None,
            warm_start_forward_level: float = 0.3,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        """Sampling.
        
        Inputs:
        - prior: torch.Tensor
            The known fixed portion of the input data. Should be in the shape of generated data.
            Use `torch.zeros((n_samples, *x_shape))` for non-prior sampling.
        - x1: torch.Tensor
            The samples from the source distribution. `None` indicates standard Gaussian samples.
        
        - n_samples: int
            The number of samples to generate.
        - sample_steps: int
            The number of sampling steps. Should be greater than 1 and less than or equal to the number of diffusion steps.
        - sample_step_schedule: Union[str, Callable]
            The schedule for the sampling steps.
        - use_ema: bool
            Whether to use the exponential moving average model.
        - temperature: float
            The temperature for sampling.
        
        - condition_cfg: Optional
            Condition for Classifier-free-guidance.
        - mask_cfg: Optional
            Mask for Classifier-guidance.
        - w_cfg: float
            Weight for Classifier-free-guidance.
        - condition_cg: Optional
            Condition for Classifier-guidance.
        - w_cg: float
            Weight for Classifier-guidance.
            
        - diffusion_x_sampling_steps: int
            The number of diffusion steps for diffusion-x sampling.
        
        - requires_grad: bool
            Whether to preserve gradients.
        - preserve_history: bool
            Whether to preserve the sampling history.
            
        Outputs:
        - x0: torch.Tensor
            Generated samples. Be in the shape of `(n_samples, *x_shape)`.
        - log: dict
            The log dictionary.
        """
        assert w_cg == 0.0 and condition_cg is None, "Rectified Flow does not support classifier-guidance."

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor):
            t_c = torch.ones_like(prior) * warm_start_forward_level
            x1 = torch.randn_like(prior) * t_c + warm_start_reference * (1 - t_c)
        else:
            if x1 is None:
                x1 = torch.randn_like(prior) * temperature
            else:
                assert prior.shape == x1.shape, "prior and x1 must have the same shape"

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        
        model = self.model if not use_ema else self.model_ema

        xt = x1.clone()
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None

        # ===================== Sampling Schedule ====================
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            final_t = warm_start_forward_level
        else:
            final_t = 1.
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    [0., final_t], sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule([0., final_t], sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):
            
            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)
            r = torch.full((n_samples,), sample_step_schedule[i - 1], dtype=torch.float32, device=self.device)
            delta_t = sample_step_schedule[i] - sample_step_schedule[i - 1]

            # velocity
            if w_cfg != 0.0 and w_cfg != 1.0 and condition_vec_cfg is not None:
                raise NotImplementedError("InsQL does not support classifier-free guidance for now. Will support in the future.")
                # repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                # vel_all = model["diffusion"](
                #     xt.repeat(*repeat_dim), t.repeat(2),
                #     torch.cat([condition_vec_cfg, torch.zeros_like(condition_vec_cfg)], 0))
                # vel_cond, vel_uncond = vel_all.chunk(2, dim=0)
                # vel = w_cfg * vel_cond + (1 - w_cfg) * vel_uncond
            elif w_cfg == 0.0 or condition_vec_cfg is None:
                raise NotImplementedError("InsQL does not support classifier-free guidance for now. Will support in the future.")
                # with torch.set_grad_enabled(requires_grad):
                #     vel = model["diffusion"](xt, t,r, None)
            else:
                with torch.set_grad_enabled(requires_grad):
                    vel = model["diffusion"](xt, t,r, condition_vec_cfg)

            # one-step update
            xt = xt - delta_t * vel

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
