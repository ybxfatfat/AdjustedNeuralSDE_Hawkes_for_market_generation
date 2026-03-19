import torch
import torch.nn as nn
import torch.nn.functional as F

class MicrostructureHawkes(nn.Module):
    def __init__(self, state_size: int, sig_channels: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        
        # ==========================================================
        # 🛡️ 核心升级：多变量 Hawkes (Multivariate Hawkes)
        # 接收 41个通道的独立Mask + 41个通道的跳跃幅度 = 82维
        # ==========================================================
        self.jump_rnn = nn.GRU(
            input_size=state_size * 2,  # <--- 从 1+state_size 改为 state_size*2
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # 强度网络现在输出 41 个独立的 lambda 强度 (每个通道预测自己的跳跃概率)
        self.intensity_net = nn.Sequential(
            nn.Linear(hidden_size + sig_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, state_size), # <--- 从 1 改为 state_size
            nn.Softplus() 
        )
        
        self.mark_net = nn.Sequential(
            nn.Linear(hidden_size + sig_channels, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, state_size)
        )

    def forward(self, jump_masks: torch.Tensor, jump_sizes: torch.Tensor, signature_context: torch.Tensor):
        batch_size, seq_len, _ = jump_masks.shape
        rnn_input = torch.cat([jump_masks, jump_sizes], dim=-1)
        rnn_out, _ = self.jump_rnn(rnn_input)
        sig_expanded = signature_context.unsqueeze(1).expand(-1, seq_len, -1)
        combined_features = torch.cat([rnn_out, sig_expanded], dim=-1)
        
        intensities = self.intensity_net(combined_features)
        predicted_marks = self.mark_net(combined_features)
        
        return intensities, predicted_marks

    def compute_loss(self, jump_masks: torch.Tensor, jump_sizes: torch.Tensor, 
                     intensities: torch.Tensor, predicted_marks: torch.Tensor, dt: float = 1.0):
        target_masks = jump_masks[:, 1:, :]
        target_sizes = jump_sizes[:, 1:, :]
        
        pred_intensities = intensities[:, :-1, :]
        pred_marks = predicted_marks[:, :-1, :]
        
        eps = 1e-8
        integral_term = pred_intensities * dt
        event_term = target_masks * torch.log(pred_intensities * dt + eps)
        
        # 多变量强度损失：对所有通道的似然求均值
        nll_intensity = (integral_term - event_term).mean()
        
        mark_error = target_masks * (pred_marks - target_sizes) ** 2
        num_jumps = target_masks.sum() + eps
        mse_mark = mark_error.sum() / num_jumps
        
        total_loss = nll_intensity + 0.1 * mse_mark
        
        return total_loss, nll_intensity.item(), mse_mark.item()
