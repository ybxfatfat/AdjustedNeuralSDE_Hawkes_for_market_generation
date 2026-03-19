import torch
import numpy as np
from scipy.stats import wasserstein_distance

class MicrostructureEvaluator:
    def __init__(self, jump_k_sigma: float = 3.0, max_lag: int = 10):
        self.jump_k_sigma = jump_k_sigma 
        self.max_lag = max_lag

    def _extract_jumps(self, paths: torch.Tensor, channel_idx: int = 0) -> np.ndarray:
        diffs = torch.diff(paths[:, :, channel_idx], dim=1).cpu().numpy()
        # 动态阈值：计算样本自身的标准差
        std_diffs = np.std(diffs, axis=1, keepdims=True)
        std_diffs = np.maximum(std_diffs, 1e-6)
        
        jumps = diffs[np.abs(diffs) > self.jump_k_sigma * std_diffs]
        return jumps

    def _compute_acf(self, paths: torch.Tensor, channel_idx: int = 0) -> np.ndarray:
        diffs = torch.diff(paths[:, :, channel_idx], dim=1)
        abs_diffs = torch.abs(diffs)
        batch_size, seq_len = abs_diffs.shape
        acf_values = np.zeros(self.max_lag)
        
        centered = abs_diffs - torch.mean(abs_diffs, dim=1, keepdim=True)
        var = torch.mean(centered ** 2, dim=1)
        
        for lag in range(1, self.max_lag + 1):
            if seq_len <= lag: break
            cov = torch.mean(centered[:, lag:] * centered[:, :-lag], dim=1)
            acf_values[lag-1] = torch.mean(cov / torch.clamp(var, min=1e-8)).item()
            
        return acf_values

    def evaluate_jump_distribution(self, real_paths: torch.Tensor, fake_paths: torch.Tensor, channel_idx: int = 0) -> float:
        real_jumps = self._extract_jumps(real_paths, channel_idx)
        fake_jumps = self._extract_jumps(fake_paths, channel_idx)
        
        # 如果真实市场和生成市场都没有发生 3-sigma 级别的跳跃，说明拟合得很完美
        if len(real_jumps) == 0 and len(fake_jumps) == 0: return 0.0
        # 如果一边有跳跃一边完全没有，给予严重的分布失真惩罚
        if len(real_jumps) == 0 or len(fake_jumps) == 0: return 999.0
            
        return wasserstein_distance(real_jumps, fake_jumps)

    def evaluate_clustering_behavior(self, real_paths: torch.Tensor, fake_paths: torch.Tensor, channel_idx: int = 0) -> float:
        return np.mean((self._compute_acf(real_paths, channel_idx) - self._compute_acf(fake_paths, channel_idx)) ** 2)

    def evaluate_intensity_reconstruction(self, real_paths: torch.Tensor, fake_paths: torch.Tensor, channel_idx: int = 0) -> float:
        real_diffs = torch.diff(real_paths[:, :, channel_idx], dim=1).abs()
        fake_diffs = torch.diff(fake_paths[:, :, channel_idx], dim=1).abs()
        
        real_std = torch.std(real_diffs, dim=1, keepdim=True).clamp(min=1e-6)
        fake_std = torch.std(fake_diffs, dim=1, keepdim=True).clamp(min=1e-6)
        
        real_counts = (real_diffs > self.jump_k_sigma * real_std).sum(dim=1).float()
        fake_counts = (fake_diffs > self.jump_k_sigma * fake_std).sum(dim=1).float()
        
        return torch.abs(real_counts.mean() - fake_counts.mean()).item()

    def run_full_evaluation(self, real_paths: torch.Tensor, fake_paths: torch.Tensor) -> dict:
        price_channel = 0
        return {
            "Jump_Dist_W1_Distance": self.evaluate_jump_distribution(real_paths, fake_paths, price_channel),
            "Clustering_ACF_MSE": self.evaluate_clustering_behavior(real_paths, fake_paths, price_channel),
            "Intensity_Mean_Error": self.evaluate_intensity_reconstruction(real_paths, fake_paths, price_channel)
        }
