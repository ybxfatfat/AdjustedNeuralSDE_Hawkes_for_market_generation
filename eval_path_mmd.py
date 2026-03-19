import torch
import torch.nn as nn
import signatory

class SignatureMMDLoss(nn.Module):
    def __init__(self, sig_depth: int = 2, kernel_type: str = 'rbf', gamma: float = 1.0):
        super().__init__()
        self.sig_depth = sig_depth
        self.kernel_type = kernel_type
        self.gamma = gamma

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == 'linear':
            return torch.matmul(x, y.t())
        elif self.kernel_type == 'rbf':
            x_norm = (x ** 2).sum(1).view(-1, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
            dist_sq = x_norm + y_norm - 2.0 * torch.matmul(x, y.t())
            dist_sq = torch.clamp(dist_sq, min=0.0)
            return torch.exp(-self.gamma * dist_sq)
        else:
            raise ValueError(f"不支持的核类型: {self.kernel_type}")

    def forward(self, real_paths: torch.Tensor, fake_paths: torch.Tensor) -> torch.Tensor:
        real_paths = real_paths.contiguous()
        fake_paths = fake_paths.contiguous()

        sig_real = signatory.signature(real_paths, depth=self.sig_depth)
        sig_fake = signatory.signature(fake_paths, depth=self.sig_depth)

        K_XX = self.compute_kernel_matrix(sig_real, sig_real)
        K_YY = self.compute_kernel_matrix(sig_fake, sig_fake)
        K_XY = self.compute_kernel_matrix(sig_real, sig_fake)

        N = sig_real.size(0)
        M = sig_fake.size(0)
        
        sum_K_XX = K_XX.sum() - torch.diag(K_XX).sum()
        sum_K_YY = K_YY.sum() - torch.diag(K_YY).sum()
        sum_K_XY = K_XY.sum() 
        
        mmd_sq = (sum_K_XX / (N * (N - 1))) + \
                 (sum_K_YY / (M * (M - 1))) - \
                 (2.0 * sum_K_XY / (N * M))
                 
        return torch.clamp(mmd_sq, min=0.0)
