from cleandiffuser.nn_diffusion import BaseNNDiffusion
from typing import Optional
import torch
import torch.nn as nn

import copy
def reset_module_parameters(module):
    for layer in module.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class NoisedQ(BaseNNDiffusion):
    def __init__(
        self, 
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        use_time_emb: bool = False,
        use_layer_norm: bool = True,
        emb_dim: int = 16, 
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None
    ):
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        
        self.use_time_emb = use_time_emb
        
        if use_time_emb:
            self.time_mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2), nn.Mish(), nn.Linear(emb_dim * 2, emb_dim))
            input_dim = obs_dim + act_dim + emb_dim
        else:
            input_dim = obs_dim + act_dim
        
        if use_layer_norm:
            self.q_model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, 1))
        else:
            self.q_model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
                nn.Linear(hidden_dim, 1))
            
        
    def forward(self, obs, noised_act, t=None):
        if self.use_time_emb:
            assert t is not None, "t must be provided when use_time_emb is True"
            emb = self.map_noise(t)
            emb = self.time_mlp(emb)
            x = torch.cat([obs, noised_act, emb], dim=-1)
        else:
            x = torch.cat([obs, noised_act], dim=-1)
        return self.q_model(x)

class EssembleNoisedQ(nn.Module):
    def __init__(self, demo_qnet, num_ensemble=10, ema_rate=0.995):
        super().__init__()
        self.qnets = nn.ModuleList([copy.deepcopy(demo_qnet) for _ in range(num_ensemble)])
        for qnet in self.qnets:
            reset_module_parameters(qnet)
        self.target_qnets = nn.ModuleList([copy.deepcopy(qnet) for qnet in self.qnets])
        self.target_qnets.requires_grad_(False).eval()
        self.ema_rate = ema_rate
        
    def forward(self, obs, noised_act, t=None, qtype='q'):
        return self.q(obs, noised_act, t, qtype)
    
    def q(self, obs, noised_act, t=None, qtype='q'):
        assert qtype in ['q', 'target_q'], "qtype must be either 'q' or 'target_q'"
        if qtype == 'q':
            q_values = torch.stack([qnet(obs, noised_act, t) for qnet in self.qnets], dim=0)
        elif qtype == 'target_q':
            q_values = torch.stack([target_qnet(obs, noised_act, t) for target_qnet in self.target_qnets], dim=0)
        return q_values

    def q_min(self, obs, noised_act, t=None, qtype='q'):
        q_values = self.q(obs, noised_act, t, qtype=qtype)
        return torch.min(q_values, dim=0).values
    
    def q_mean(self, obs, noised_act, t=None, qtype='q'):
        q_values = self.q(obs, noised_act, t, qtype=qtype)
        return torch.mean(q_values, dim=0)

    def q_std(self, obs, noised_act, t=None, qtype='q'):
        q_values = self.q(obs, noised_act, t, qtype=qtype)
        return torch.std(q_values, dim=0)
    
    def update_target(self):
        for q, target_q in zip(self.qnets, self.target_qnets):
            for q_param, target_q_param in zip(q.parameters(), target_q.parameters()):
                target_q_param.data = target_q_param.data * self.ema_rate + q_param.data * (1 - self.ema_rate)
