from typing import Optional

import torch
import torch.nn as nn

from cleandiffuser.utils import JoshMLP
from cleandiffuser.nn_diffusion import BaseNNDiffusion

from torch.distributions.normal import Normal

import torch.nn.functional as F


class VAEDistillMLP(BaseNNDiffusion):
    
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            policy_in_dim: int,
            planner_in_dim: int,
            hidden_dim: int = 256,
            emb_dim: int = 16,
            policy_hidden_num: int = 2,
            planner_hidden_num: int = 1,
            decoder_hidden_num: int = 1,
            nn_activation: nn.Module = nn.Mish(),
            latent_dim: int = 32,
            timestep_emb_type: str = "positional",
            timestep_emb_params: Optional[dict] = None
    ):
        
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        assert obs_dim == policy_in_dim, "obs_dim should be equal to policy_in_dim"
        
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim))
        
        # policy side
        self.policy_encoder = JoshMLP(
            dims=[obs_dim + act_dim + emb_dim] + [hidden_dim] * policy_hidden_num,
            activation=nn_activation,
            out_activation=nn_activation,
        )
        
        ## [Policy] mean and logvar of latent variable
        self.policy_z_mean = JoshMLP(
            dims=[hidden_dim, latent_dim],
            activation=nn_activation,
            out_activation=nn.Identity(),
        )
        self.policy_z_logstd = JoshMLP(
            dims=[hidden_dim, latent_dim],
            activation=nn_activation,
            out_activation=nn.Identity(),
        )
        
        # planner side
        self.planner_encoder = JoshMLP(
            dims=[planner_in_dim]+[hidden_dim]*planner_hidden_num,
            activation=nn_activation,
            out_activation=nn_activation,
        )
        
        ## [Planner] mean and logvar of latent variable
        self.planner_z_mean = JoshMLP(
            dims=[hidden_dim, latent_dim],
            activation=nn_activation,
            out_activation=nn.Identity(),
        )
        self.planner_z_logstd = JoshMLP(
            dims=[hidden_dim, latent_dim],
            activation=nn_activation,
            out_activation=nn.Identity(),
        )
        
        # decoder
        self.decoder = JoshMLP(
            dims=[latent_dim]+[hidden_dim]*decoder_hidden_num+[act_dim],
            activation=nn_activation,
            out_activation=nn.Identity(),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_policy_z(self, x, noise, condition):
        t = self.time_mlp(self.map_noise(noise))
        x = torch.cat([x, t, condition], -1)
        policy_latent = self.policy_encoder(x)
        policy_z_mean = self.policy_z_mean(policy_latent)
        policy_z_std = F.softplus(self.policy_z_logstd(policy_latent))
        return policy_z_mean, policy_z_std
    
    def get_planner_z(self, x):
        planner_latent = self.planner_encoder(x)
        planner_z_mean = self.planner_z_mean(planner_latent)
        planner_z_std = F.softplus(self.planner_z_logstd(planner_latent))
        return planner_z_mean, planner_z_std

    def policy_forward(self, x, noise, condition, deterministic=False):
        """
        Input:
            x:          (b, act_dim)
            noise:      (b, )
            condition:  (b, obs_dim)

        Output:
            y:          (b, act_dim)
        """
        policy_z_mean, policy_z_std = self.get_policy_z(x, noise, condition)
        
        if deterministic:
            policy_z = policy_z_mean
        else:
            dist = Normal(policy_z_mean, policy_z_std)
            policy_z = dist.rsample()
            
        return self.decoder(policy_z)
    
    def planner_forward(self, x, deterministic=False):
        """
        Input:
            x:          (b, hidden_dim * 2) or (b, obs_dim * 2)

        Output:
            y:          (b, latent_dim)
        """
        planner_z_mean, planner_z_std = self.get_planner_z(x)
        
        if deterministic:
            planner_z = planner_z_mean
        else:
            dist = Normal(planner_z_mean, planner_z_std)
            planner_z = dist.rsample()
            
        return self.decoder(planner_z)
    
    def forward(self, x):
        return self.planner_forward(x)