import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ==========================================
# 1. 纯 PyTorch: Black-Scholes 公式与微观成本
# ==========================================
def black_scholes_call_delta(S: torch.Tensor, K: float, T: float, t: torch.Tensor, r: float = 0.0, sigma: float = 0.2):
    """
    纯 PyTorch 实现的 Black-Scholes 欧式看涨期权 Delta 计算。
    公式: N(d1), 其中 d1 = (ln(S/K) + (r + sigma^2 / 2) * tau) / (sigma * sqrt(tau))
    """
    tau = T - t
    # 防止到期时 tau 为 0 导致除零错误
    tau = torch.clamp(tau, min=1e-8)
    
    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * torch.sqrt(tau))
    
    # 标准正态分布的 CDF
    normal = torch.distributions.Normal(0, 1)
    delta = normal.cdf(d1)
    return delta

class MicrostructureCost(nn.Module):
    def __init__(self, impact_lambda: float = 0.01):
        super().__init__()
        self.impact_lambda = impact_lambda

    def forward(self, delta_action: torch.Tensor, spread: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
        """摩擦成本 = 跨越价差 + 消耗流动性带来的非线性冲击"""
        abs_action = torch.abs(delta_action)
        spread_cost = (spread / 2.0) * abs_action
        eps = 1e-8
        impact_cost = self.impact_lambda * (delta_action ** 2) / (volume + eps)
        return spread_cost + impact_cost

# ==========================================
# 2. 纯 PyTorch: Deep Hedger 神经网络
# ==========================================
class PurePyTorchHedger(nn.Module):
    def __init__(self):
        """
        多层感知机 (MLP) 策略网络。
        输入: 对数在值程度 (Log-Moneyness), 剩余到期时间 (Time-to-Maturity)
        输出: 对冲比例 Delta (对于看涨期权，范围在 0 到 1 之间)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 看涨期权的 Delta 严格在 (0, 1) 之间
        )
        
    def forward(self, log_moneyness: torch.Tensor, time_to_maturity: torch.Tensor) -> torch.Tensor:
        # 拼接特征 [Batch, Time, 2]
        x = torch.stack([log_moneyness, time_to_maturity], dim=-1)
        # 输出形状 [Batch, Time]
        return self.net(x).squeeze(-1)

# ==========================================
# 3. 深度对冲训练与评估主类
# ==========================================
class MicrostructureHedgingEnv:
    def __init__(self, dt: float = 1/250/100, strike: float = 1.0, cost_lambda: float = 0.01):
        self.dt = dt
        self.strike = strike
        self.cost_model = MicrostructureCost(impact_lambda=cost_lambda)
        # 假设历史波动率，用于 BS 基准对照
        self.bs_sigma = 0.05 

    def calculate_features(self, prices: torch.Tensor):
        """计算神经网络和 BS 需要的状态特征"""
        batch_size, seq_len = prices.shape
        device = prices.device
        
        # 剩余时间: 从 T 倒数到 0
        T_total = seq_len * self.dt
        t_steps = torch.arange(0, seq_len, device=device) * self.dt
        time_to_maturity = T_total - t_steps
        time_to_maturity = time_to_maturity.unsqueeze(0).expand(batch_size, -1)
        
        # 对数在值程度: ln(S_t / K)
        log_moneyness = torch.log(prices / self.strike)
        
        return log_moneyness, time_to_maturity

    def compute_path_pnl(self, prices: torch.Tensor, spreads: torch.Tensor, volumes: torch.Tensor, actions: torch.Tensor):
        """计算整条路径的最终盈亏 (Terminal PnL)"""
        # 1. 期权卖方的最终赔付 (Payoff)
        terminal_prices = prices[:, -1]
        payoff = torch.relu(terminal_prices - self.strike)
        
        # 2. 标的资产交易的持仓收益 (积分 Delta * dS)
        price_changes = torch.diff(prices, dim=1)
        action_shifted = actions[:, :-1] # t 时刻的动作只能吃 t 到 t+1 的价格变化
        trading_pnl = torch.sum(action_shifted * price_changes, dim=1)
        
        # 3. 微观结构摩擦成本
        delta_action = torch.diff(actions, dim=1, prepend=torch.zeros_like(actions[:, :1]))
        costs = self.cost_model(delta_action, spreads, volumes)
        total_cost = torch.sum(costs, dim=1)
        
        # 最终 PnL = 对冲收益 - 期权赔付 - 交易成本
        return trading_pnl - payoff - total_cost

    def train_and_evaluate(self, train_data: dict, test_data: dict, epochs: int = 50):
        print("构建纯 PyTorch 训练环境...")
        device = train_data['prices'].device
        
        train_prices = train_data['prices']
        train_spreads = train_data['spreads']
        train_volumes = train_data['volumes']
        
        test_prices = test_data['prices']
        test_spreads = test_data['spreads']
        test_volumes = test_data['volumes']

        # 初始化模型
        deep_hedger = PurePyTorchHedger().to(device)
        optimizer = torch.optim.Adam(deep_hedger.parameters(), lr=0.005)

        print("开始训练 Deep Hedger (适配高频交易成本)...")
        deep_hedger.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 计算特征并获取网络动作
            log_m, ttm = self.calculate_features(train_prices)
            actions = deep_hedger(log_m, ttm)
            
            # 计算 PnL 与 Loss (均方误差，即最小化 PnL 的方差)
            pnl = self.compute_path_pnl(train_prices, train_spreads, train_volumes, actions)
            loss = torch.mean(pnl ** 2)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Hedging Loss (MSE): {loss.item():.6f}")

        print("在生成的高频路径上评估模型...")
        deep_hedger.eval()
        with torch.no_grad():
            # 1. 评估 Deep Hedger
            test_log_m, test_ttm = self.calculate_features(test_prices)
            action_deep = deep_hedger(test_log_m, test_ttm)
            pnl_deep = self.compute_path_pnl(test_prices, test_spreads, test_volumes, action_deep)
            
            # 2. 评估 Black-Scholes 基准
            # 为了公平对比，BS 也会遭受相同的微观结构成本惩罚
            action_bs = black_scholes_call_delta(
                S=test_prices, K=self.strike, T=test_prices.shape[1] * self.dt, 
                t=test_ttm[0, 0] - test_ttm, sigma=self.bs_sigma
            )
            pnl_bs = self.compute_path_pnl(test_prices, test_spreads, test_volumes, action_bs)

        return pnl_deep, pnl_bs, action_deep, action_bs, test_prices

# ==========================================
# 4. 高级可视化
# ==========================================
def plot_hedging_results(pnl_deep: torch.Tensor, pnl_bs: torch.Tensor, 
                         action_deep: torch.Tensor, action_bs: torch.Tensor, 
                         price_path: torch.Tensor, save_path: str = "hedging_results.png"):
    pnl_deep_np = pnl_deep.cpu().numpy()
    pnl_bs_np = pnl_bs.cpu().numpy()
    
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = plt.subplot(1, 2, 1)
    sns.kdeplot(pnl_bs_np, fill=True, color="crimson", label="Black-Scholes (Delta)", alpha=0.5, ax=ax1)
    sns.kdeplot(pnl_deep_np, fill=True, color="royalblue", label="Deep Hedging (AI)", alpha=0.5, ax=ax1)
    
    ax1.axvline(np.mean(pnl_bs_np), color="crimson", linestyle="--", alpha=0.8)
    ax1.axvline(np.mean(pnl_deep_np), color="royalblue", linestyle="--", alpha=0.8)
    
    ax1.set_title("Terminal PnL Distribution (Including Microstructure Costs)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Profit & Loss (PnL)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper left")
    
    ax2 = plt.subplot(1, 2, 2)
    sample_idx = 0 
    time_steps = np.arange(action_deep.shape[1])
    
    l1 = ax2.plot(time_steps, action_bs[sample_idx].cpu().numpy(), color="crimson", label="BS Delta", linewidth=2)
    l2 = ax2.plot(time_steps, action_deep[sample_idx].cpu().numpy(), color="royalblue", label="Deep Hedger", linewidth=2, linestyle="-.")
    ax2.set_ylabel("Hedge Position (Delta)", color="black")
    ax2.set_xlabel("Time Steps (High-Frequency Ticks)")
    
    ax3 = ax2.twinx()
    l3 = ax3.plot(time_steps, price_path[sample_idx].cpu().numpy(), color="gray", alpha=0.4, label="Underlying Mid-price")
    ax3.set_ylabel("Asset Price", color="gray")
    
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right")
    
    ax2.set_title("Hedging Trajectory on a Jump-Diffusion Path", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")
    plt.show()
