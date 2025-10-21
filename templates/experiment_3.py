# 实验 3：差分隐私对比测试（代码示例）
#
# 目标：比较“未使用差分隐私（明文聚合）”与“使用差分隐私（梯度裁剪 + 高斯噪声）”后，
# 全局模型在测试集上的推理准确率差异。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import random
from experiment_3_plot import main_plotting

# Opacus
try:
    from opacus import PrivacyEngine

    OPACUS_AVAILABLE = True
except ImportError:
    print("Opacus 未安装，将使用手动实现的差分隐私")
    OPACUS_AVAILABLE = False


# 简单 CNN（用于医学影像可替换为实际网络）
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def synthetic_dataset(num_samples=1000, img_size=32):
    # 生成合成数据用于演示
    X = np.random.randn(num_samples, 1, img_size, img_size).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.int64)
    return X, y


def train_one_round_with_dp(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """手动实现差分隐私训练"""
    model.train()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()


def train_one_round_standard(model, dataloader, optimizer, criterion, device):
    """标准训练（无DP）"""
    model.train()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()


def add_noise_to_parameters(model, noise_multiplier=1.0):
    """手动向模型参数添加高斯噪声"""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_multiplier * 0.01
            param.add_(noise)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def run_experiment_with_opacus(use_dp=False, dp_noise_multiplier=1.0, max_grad_norm=1.0):
    """使用Opacus库的DP实现"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    X, y = synthetic_dataset(1000, img_size=32)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_set, test_set = random_split(dataset, [800, 200])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    if use_dp and OPACUS_AVAILABLE:
        # 新版本Opacus的API
        privacy_engine = PrivacyEngine()

        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=dp_noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

    # 训练若干 epoch 模拟联邦本地训练
    for epoch in range(5):
        if use_dp and OPACUS_AVAILABLE:
            train_one_round_standard(model, train_loader, optimizer, criterion, device)
        else:
            train_one_round_standard(model, train_loader, optimizer, criterion, device)

    acc = evaluate(model, test_loader, device)

    if use_dp and OPACUS_AVAILABLE:
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"隐私预算: ε = {epsilon:.3f} (δ = 1e-5)")

    return acc


def run_experiment_manual_dp(use_dp=False, dp_noise_multiplier=1.0, max_grad_norm=1.0):
    """手动实现差分隐私"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    X, y = synthetic_dataset(1000, img_size=32)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_set, test_set = random_split(dataset, [800, 200])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 训练若干 epoch 模拟联邦本地训练
    for epoch in range(5):
        if use_dp:
            train_one_round_with_dp(model, train_loader, optimizer, criterion, device, max_grad_norm)
            # 在每轮训练后添加噪声
            add_noise_to_parameters(model, dp_noise_multiplier)
        else:
            train_one_round_standard(model, train_loader, optimizer, criterion, device)

    acc = evaluate(model, test_loader, device)
    return acc


def main():
    print("=== 差分隐私对比实验 ===")

    # 方法选择
    if OPACUS_AVAILABLE:
        print("检测到Opacus库，使用Opacus实现")
        experiment_func = run_experiment_with_opacus
    else:
        print("未检测到Opacus库，使用手动实现")
        experiment_func = run_experiment_manual_dp

    # 多次实验取平均值
    num_trials = 3
    results = {
        'no_dp': [],
        'dp_low': [],
        'dp_high': []
    }

    for trial in range(num_trials):
        print(f"\n--- 实验 {trial + 1}/{num_trials} ---")

        # 不使用 DP
        acc_no_dp = experiment_func(use_dp=False)
        results['no_dp'].append(acc_no_dp)
        print(f"不使用差分隐私准确率：{acc_no_dp:.4f}")

        # 使用 DP (噪声较小)
        acc_dp = experiment_func(use_dp=True, dp_noise_multiplier=1.0, max_grad_norm=1.0)
        results['dp_low'].append(acc_dp)
        print(f"使用差分隐私（noise=1.0）准确率：{acc_dp:.4f}")

        # 使用更强 DP (噪声更大)
        acc_dp2 = experiment_func(use_dp=True, dp_noise_multiplier=2.0, max_grad_norm=1.0)
        results['dp_high'].append(acc_dp2)
        print(f"使用差分隐私（noise=2.0）准确率：{acc_dp2:.4f}")

    # 输出统计结果
    # 在您的主实验代码最后添加：

    # 假设您的实验结果存储在 results 字典中
    # results = {
    #     'no_dp': [acc1, acc2, acc3, ...],
    #     'dp_low': [acc1, acc2, acc3, ...],
    #     'dp_high': [acc1, acc2, acc3, ...]
    # }

    # 生成图表
     # 假设您将绘图代码保存为 dp_plotting.py
    main_plotting(results)
    print("\n=== 最终结果统计 ===")
    print(f"无DP准确率: {np.mean(results['no_dp']):.4f} ± {np.std(results['no_dp']):.4f}")
    print(f"低噪声DP准确率: {np.mean(results['dp_low']):.4f} ± {np.std(results['dp_low']):.4f}")
    print(f"高噪声DP准确率: {np.mean(results['dp_high']):.4f} ± {np.std(results['dp_high']):.4f}")

    # 计算准确率下降百分比
    base_acc = np.mean(results['no_dp'])
    dp_low_acc = np.mean(results['dp_low'])
    dp_high_acc = np.mean(results['dp_high'])

    print(f"\n准确率下降百分比:")
    print(f"低噪声DP: {((base_acc - dp_low_acc) / base_acc * 100):.2f}%")
    print(f"高噪声DP: {((base_acc - dp_high_acc) / base_acc * 100):.2f}%")




if __name__ == "__main__":
    # 设置随机种子以便复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    main()