import torch
import torchcde
import signatory
from torch.utils.data import Dataset, DataLoader

class UniversalMicrostructureDataset(Dataset):
    def __init__(self, pt_file_path: str, sig_depth: int = 2, jump_k_sigma: float = 3.0):
        """
        jump_k_sigma: 动态跳跃检验的倍数。3.0 表示超过 3 倍局部标准差才被认定为跳跃。
        """
        super().__init__()
        data_dict = torch.load(pt_file_path, weights_only=False)
        self.raw_data = data_dict['original_data']  
        self.coeffs = data_dict['coeffs']           
        self.labels = data_dict['labels']
        self.num_samples, self.seq_len, self.channels = self.raw_data.shape
        self.sig_depth = sig_depth
        self.jump_k_sigma = jump_k_sigma
        
        print(f"初始化数据管道：总样本数 {self.num_samples}，使用 {self.jump_k_sigma}-Sigma 动态跳跃检验...")
        self.clean_data, self.signatures, self.jump_masks, self.jump_sizes = self._preprocess_all()

    def _preprocess_all(self):
        device = self.raw_data.device
        chunk_size = 2000 
        
        all_clean_data, all_signatures, all_jump_masks, all_jump_sizes = [], [], [], []
        
        for i in range(0, self.num_samples, chunk_size):
            end_idx = min(i + chunk_size, self.num_samples)
            raw_chunk = self.raw_data[i:end_idx]
            coeff_chunk = self.coeffs[i:end_idx]
            
            X = torchcde.CubicSpline(coeff_chunk)
            t_grid = torch.linspace(X.interval[0], X.interval[1], self.seq_len, device=device)
            clean_chunk = torch.where(torch.isnan(raw_chunk), X.evaluate(t_grid), raw_chunk).contiguous()
            
            with torch.no_grad():
                sig_chunk = signatory.signature(clean_chunk, depth=self.sig_depth).detach()
                
            time_diff = torch.diff(clean_chunk, dim=1)
            time_diff_padded = torch.cat([torch.zeros(end_idx - i, 1, self.channels, device=device), time_diff], dim=1)
            
            # ==========================================================
            # 📊 数据驱动：计算局部波动率，进行 k-Sigma 跳跃检验
            # ==========================================================
            chunk_std = torch.std(time_diff_padded, dim=1, keepdim=True).clamp(min=1e-6)
            
            # 判定跳跃：绝对变化量 > k * 局部标准差
            jump_masks = (torch.abs(time_diff_padded) > self.jump_k_sigma * chunk_std).float()
            jump_sizes = time_diff_padded * jump_masks 
            
            all_clean_data.append(clean_chunk.cpu())
            all_signatures.append(sig_chunk.cpu())
            all_jump_masks.append(jump_masks.cpu())
            all_jump_sizes.append(jump_sizes.cpu())
            
        return torch.cat(all_clean_data), torch.cat(all_signatures), torch.cat(all_jump_masks), torch.cat(all_jump_sizes)

    def __len__(self) -> int: return self.num_samples
    def __getitem__(self, idx: int) -> dict:
        return {'coeffs': self.coeffs[idx], 'signature_context': self.signatures[idx],
                'clean_path': self.clean_data[idx], 'jump_mask': self.jump_masks[idx],
                'jump_size': self.jump_sizes[idx], 'label': self.labels[idx]}

def get_universal_dataloader(pt_file_path: str, batch_size: int = 32, shuffle: bool = True):
    return DataLoader(UniversalMicrostructureDataset(pt_file_path), batch_size=batch_size, shuffle=shuffle)
