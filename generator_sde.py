import torch
import torch.nn as nn
import torchsde

class MicrostructureSDE(torchsde.SDEIto):
    def __init__(self, state_size: int, sig_channels: int, hidden_size: int = 64, num_layers: int = 3):
        super().__init__(noise_type="diagonal")
        self.state_size = state_size
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.current_sig_context = None

        drift_layers = [nn.Linear(state_size + sig_channels, hidden_size), nn.GELU()]
        for _ in range(num_layers - 1):
            drift_layers.append(nn.Linear(hidden_size, hidden_size))
            drift_layers.append(nn.GELU())
        drift_layers.append(nn.Linear(hidden_size, state_size))
        self.drift_net = nn.Sequential(*drift_layers)

        diff_layers = [nn.Linear(state_size + sig_channels, hidden_size), nn.GELU()]
        for _ in range(num_layers - 1):
            diff_layers.append(nn.Linear(hidden_size, hidden_size))
            diff_layers.append(nn.GELU())
        diff_layers.append(nn.Linear(hidden_size, state_size))
        diff_layers.append(nn.Sigmoid()) 
        self.diffusion_net = nn.Sequential(*diff_layers)

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_cond = torch.cat([y, self.current_sig_context], dim=-1)
        return self.drift_net(y_cond)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_cond = torch.cat([y, self.current_sig_context], dim=-1)
        min_volatility = 1e-3
        return self.diffusion_net(y_cond) + min_volatility

class SDEGenerator(nn.Module):
    def __init__(self, state_size: int, sig_channels: int, hidden_size: int = 64):
        super().__init__()
        self.sde = MicrostructureSDE(state_size, sig_channels, hidden_size)
        self.initial_state_net = nn.Sequential(
            nn.Linear(sig_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, state_size)
        )

    def forward(self, signature_context: torch.Tensor, last_observed_state: torch.Tensor, 
                future_steps: int, dt: float = 0.01) -> torch.Tensor:
        batch_size = signature_context.size(0)
        device = signature_context.device
        self.sde.current_sig_context = signature_context
        y0 = last_observed_state
        ts = torch.linspace(0, future_steps * dt, future_steps, device=device)

        generated_paths = torchsde.sdeint_adjoint(
            sde=self.sde,
            y0=y0,
            ts=ts,
            method='euler',
            options=dict(step_size=dt) 
        )
        return generated_paths.transpose(0, 1)
