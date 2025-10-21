import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import tenseal as ts
import pandas as pd
import time
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class FeatureExtractor(nn.Module):
    """特征提取器 - 自定义CNN特征提取器"""

    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    """分类器网络"""

    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SyntheticMedicalDataset(Dataset):
    """合成医学图像数据集（单通道）"""

    def __init__(self, num_samples=1000, resolution=(224, 224), num_classes=2):
        self.num_samples = num_samples
        self.resolution = resolution
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成合成医学图像（模拟CT/MRI单通道）
        np.random.seed(idx)
        image = np.random.randn(1, self.resolution[0], self.resolution[1]).astype(np.float32) * 0.5
        label = np.random.randint(0, self.num_classes)
        return torch.from_numpy(image), label


class HomomorphicExperiment:
    def __init__(self, feature_dim=512, num_classes=2):
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 初始化模型（单通道输入）
        self.feature_extractor = FeatureExtractor(in_channels=1)
        self.classifier = Classifier(feature_dim, num_classes)

        # 初始化同态加密上下文
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[50, 30, 30, 30, 50]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** 30
        self.context.generate_relin_keys()

        # 模拟训练好的权重
        self._initialize_weights()

        # 计算特征提取器输出维度
        self._calculate_feature_dim()

    def _initialize_weights(self):
        """初始化模型权重"""
        torch.manual_seed(42)
        for module in [self.feature_extractor, self.classifier]:
            for param in module.parameters():
                if len(param.shape) > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 0.1)

    def _calculate_feature_dim(self):
        """计算特征提取器的输出维度"""
        with torch.no_grad():
            # 使用一个虚拟输入来计算特征维度
            dummy_input = torch.randn(1, 1, 224, 224)
            features = self.feature_extractor(dummy_input)
            self.actual_feature_dim = features.view(1, -1).shape[1]
            print(f"特征提取器输出维度: {self.actual_feature_dim}")

    def scheme_a_full_encryption(self, image):
        """方案A：完整网络同态加密（模拟）"""
        # 客户端：加密整个图像
        client_start = time.time()
        encrypted_image = ts.ckks_vector(self.context, image.flatten().tolist())
        client_encrypt_time = time.time() - client_start

        # 服务器：模拟完整网络同态推理
        server_start = time.time()
        # 计算总参数数量
        total_params = sum(p.numel() for p in self.feature_extractor.parameters()) + \
                       sum(p.numel() for p in self.classifier.parameters())
        # 模拟计算时间（基于参数数量和图像大小）
        computation_factor = total_params * len(image.flatten()) * 1e-9
        time.sleep(computation_factor)
        server_compute_time = time.time() - server_start

        # 客户端：解密结果
        client_start = time.time()
        decrypted_result = np.array(encrypted_image.decrypt()[:self.num_classes])
        client_decrypt_time = time.time() - client_start

        total_time = client_encrypt_time + server_compute_time + client_decrypt_time

        # 通信大小：加密后的图像大小
        communication_size = len(image.flatten()) * 8  # 估计每个元素8字节

        return total_time, communication_size, decrypted_result

    def scheme_b_feature_encryption(self, image):
        """方案B：特征加密（解耦方案）"""
        # 客户端：明文特征提取 + 加密特征向量
        client_start = time.time()

        # 特征提取（明文）
        with torch.no_grad():
            image_tensor = torch.from_numpy(image).unsqueeze(0)
            features = self.feature_extractor(image_tensor)
            features_flat = features.view(1, -1).numpy().flatten()

            # 如果实际特征维度与设定不同，进行调整
            if len(features_flat) != self.feature_dim:
                if len(features_flat) > self.feature_dim:
                    features_flat = features_flat[:self.feature_dim]
                else:
                    # 填充到设定维度
                    padded = np.zeros(self.feature_dim)
                    padded[:len(features_flat)] = features_flat
                    features_flat = padded

        # 加密特征向量
        encrypted_features = ts.ckks_vector(self.context, features_flat.tolist())
        client_encrypt_time = time.time() - client_start

        # 服务器：密文分类器推理
        server_start = time.time()
        with torch.no_grad():
            # 提取分类器权重
            fc1_weight = self.classifier.fc1.weight.data.numpy().T
            fc1_bias = self.classifier.fc1.bias.data.numpy()
            fc2_weight = self.classifier.fc2.weight.data.numpy().T
            fc2_bias = self.classifier.fc2.bias.data.numpy()
            fc3_weight = self.classifier.fc3.weight.data.numpy().T
            fc3_bias = self.classifier.fc3.bias.data.numpy()

        # 同态计算
        result = encrypted_features.mm(fc1_weight) + fc1_bias
        result = result.mm(fc2_weight) + fc2_bias
        result = result.mm(fc3_weight) + fc3_bias
        server_compute_time = time.time() - server_start

        # 客户端：解密结果
        client_start = time.time()
        decrypted_result = np.array(result.decrypt())
        client_decrypt_time = time.time() - client_start

        total_time = client_encrypt_time + server_compute_time + client_decrypt_time

        # 通信大小：加密后的特征向量大小
        communication_size = len(features_flat) * 8  # 估计每个元素8字节

        return total_time, communication_size, decrypted_result


def run_experiment(resolutions, num_trials=30):
    """运行实验"""
    results = []

    for resolution in resolutions:
        print(f"正在测试分辨率: {resolution[0]}x{resolution[1]}")

        # 创建数据集
        dataset = SyntheticMedicalDataset(num_samples=num_trials, resolution=resolution)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # 初始化实验
        experiment = HomomorphicExperiment(feature_dim=512)

        scheme_a_times = []
        scheme_a_sizes = []
        scheme_b_times = []
        scheme_b_sizes = []

        for i, (image, label) in enumerate(dataloader):
            image_np = image.squeeze().numpy()

            if i >= num_trials:
                break

            # 方案A测试
            time_a, size_a, _ = experiment.scheme_a_full_encryption(image_np)
            scheme_a_times.append(time_a)
            scheme_a_sizes.append(size_a)

            # 方案B测试
            time_b, size_b, _ = experiment.scheme_b_feature_encryption(image_np)
            scheme_b_times.append(time_b)
            scheme_b_sizes.append(size_b)

            if (i + 1) % 10 == 0:
                print(f"  完成 {i + 1}/{num_trials} 次试验")

        # 计算统计量
        for scheme, times, sizes in [('方案A', scheme_a_times, scheme_a_sizes),
                                     ('方案B', scheme_b_times, scheme_b_sizes)]:
            mean_time = np.mean(times)
            std_time = np.std(times)
            ci_time = 1.96 * std_time / np.sqrt(len(times))

            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            ci_size = 1.96 * std_size / np.sqrt(len(sizes))

            results.append({
                '分辨率': f"{resolution[0]}x{resolution[1]}",
                '方案': scheme,
                '平均时间_秒': mean_time,
                '时间标准差': std_time,
                '时间置信区间': ci_time,
                '平均通信大小_字节': mean_size,
                '通信大小标准差': std_size,
                '通信大小置信区间': ci_size,
                '试验次数': num_trials
            })

    return pd.DataFrame(results)


def statistical_analysis(df):
    """统计分析"""
    print("\n" + "=" * 80)
    print("统计分析结果")
    print("=" * 80)

    resolutions = df['分辨率'].unique()

    for resolution in resolutions:
        resolution_data = df[df['分辨率'] == resolution]
        scheme_a = resolution_data[resolution_data['方案'] == '方案A']
        scheme_b = resolution_data[resolution_data['方案'] == '方案B']

        # 时间t检验（使用模拟的原始数据）
        time_a_mean = scheme_a['平均时间_秒'].iloc[0]
        time_b_mean = scheme_b['平均时间_秒'].iloc[0]
        time_a_std = scheme_a['时间标准差'].iloc[0]
        time_b_std = scheme_b['时间标准差'].iloc[0]

        # 计算t统计量
        n = scheme_a['试验次数'].iloc[0]
        t_stat_time = (time_a_mean - time_b_mean) / np.sqrt((time_a_std ** 2 + time_b_std ** 2) / n)

        # 通信大小t检验
        size_a_mean = scheme_a['平均通信大小_字节'].iloc[0]
        size_b_mean = scheme_b['平均通信大小_字节'].iloc[0]
        size_a_std = scheme_a['通信大小标准差'].iloc[0]
        size_b_std = scheme_b['通信大小标准差'].iloc[0]

        t_stat_size = (size_a_mean - size_b_mean) / np.sqrt((size_a_std ** 2 + size_b_std ** 2) / n)

        # 计算p值
        p_value_time = 2 * (1 - stats.t.cdf(abs(t_stat_time), 2 * n - 2))
        p_value_size = 2 * (1 - stats.t.cdf(abs(t_stat_size), 2 * n - 2))

        print(f"\n分辨率 {resolution}:")
        print(
            f"  时间差异显著性: t={t_stat_time:.3f}, p={p_value_time:.6f} {'***' if p_value_time < 0.001 else '**' if p_value_time < 0.01 else '*' if p_value_time < 0.05 else '不显著'}")
        print(
            f"  通信大小差异显著性: t={t_stat_size:.3f}, p={p_value_size:.6f} {'***' if p_value_size < 0.001 else '**' if p_value_size < 0.01 else '*' if p_value_size < 0.05 else '不显著'}")

        # 计算优化百分比
        time_improvement = (time_a_mean - time_b_mean) / time_a_mean * 100
        size_improvement = (size_a_mean - size_b_mean) / size_a_mean * 100

        print(f"  时间优化: {time_improvement:.1f}%")
        print(f"  通信优化: {size_improvement:.1f}%")


def plot_results(df):
    """绘制结果图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 推理时间对比
    resolutions = df['分辨率'].unique()
    x = np.arange(len(resolutions))
    width = 0.35

    for i, resolution in enumerate(resolutions):
        res_data = df[df['分辨率'] == resolution]
        scheme_a = res_data[res_data['方案'] == '方案A']
        scheme_b = res_data[res_data['方案'] == '方案B']

        ax1.bar(i - width / 2, scheme_a['平均时间_秒'].iloc[0], width,
                label='方案A' if i == 0 else "", color='red', alpha=0.7,
                yerr=scheme_a['时间置信区间'].iloc[0], capsize=5)
        ax1.bar(i + width / 2, scheme_b['平均时间_秒'].iloc[0], width,
                label='方案B' if i == 0 else "", color='blue', alpha=0.7,
                yerr=scheme_b['时间置信区间'].iloc[0], capsize=5)

    ax1.set_xlabel('图像分辨率')
    ax1.set_ylabel('推理时间 (秒)')
    ax1.set_title('同态加密推理时间对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(resolutions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 通信大小对比
    for i, resolution in enumerate(resolutions):
        res_data = df[df['分辨率'] == resolution]
        scheme_a = res_data[res_data['方案'] == '方案A']
        scheme_b = res_data[res_data['方案'] == '方案B']

        ax2.bar(i - width / 2, scheme_a['平均通信大小_字节'].iloc[0] / 1024, width,
                color='red', alpha=0.7, yerr=scheme_a['通信大小置信区间'].iloc[0] / 1024, capsize=5)
        ax2.bar(i + width / 2, scheme_b['平均通信大小_字节'].iloc[0] / 1024, width,
                color='blue', alpha=0.7, yerr=scheme_b['通信大小置信区间'].iloc[0] / 1024, capsize=5)

    ax2.set_xlabel('图像分辨率')
    ax2.set_ylabel('通信大小 (KB)')
    ax2.set_title('通信开销对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(resolutions)
    ax2.grid(True, alpha=0.3)

    # 3. 优化百分比
    time_improvements = []
    size_improvements = []

    for resolution in resolutions:
        res_data = df[df['分辨率'] == resolution]
        scheme_a = res_data[res_data['方案'] == '方案A']
        scheme_b = res_data[res_data['方案'] == '方案B']

        time_imp = (scheme_a['平均时间_秒'].iloc[0] - scheme_b['平均时间_秒'].iloc[0]) / scheme_a['平均时间_秒'].iloc[
            0] * 100
        size_imp = (scheme_a['平均通信大小_字节'].iloc[0] - scheme_b['平均通信大小_字节'].iloc[0]) / \
                   scheme_a['平均通信大小_字节'].iloc[0] * 100

        time_improvements.append(time_imp)
        size_improvements.append(size_imp)

    ax3.bar(x - width / 2, time_improvements, width, label='时间优化', color='green', alpha=0.7)
    ax3.bar(x + width / 2, size_improvements, width, label='通信优化', color='orange', alpha=0.7)

    ax3.set_xlabel('图像分辨率')
    ax3.set_ylabel('优化百分比 (%)')
    ax3.set_title('解耦方案优化效果')
    ax3.set_xticks(x)
    ax3.set_xticklabels(resolutions)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 散点图显示关系
    resolutions_numeric = [int(res.split('x')[0]) for res in resolutions]
    ax4.scatter(resolutions_numeric, time_improvements, s=100, alpha=0.7, label='时间优化')
    ax4.scatter(resolutions_numeric, size_improvements, s=100, alpha=0.7, label='通信优化')

    # 添加趋势线
    if len(resolutions_numeric) > 1:
        z_time = np.polyfit(resolutions_numeric, time_improvements, 1)
        z_size = np.polyfit(resolutions_numeric, size_improvements, 1)

        x_range = np.linspace(min(resolutions_numeric), max(resolutions_numeric), 100)
        ax4.plot(x_range, np.poly1d(z_time)(x_range), 'g--', alpha=0.7)
        ax4.plot(x_range, np.poly1d(z_size)(x_range), 'orange', linestyle='--', alpha=0.7)

    ax4.set_xlabel('分辨率 (像素)')
    ax4.set_ylabel('优化百分比 (%)')
    ax4.set_title('分辨率与优化效果关系')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('同态加密实验一结果.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=== 实验一：同态加密方案对比 ===")
    print("方案A: 完整网络加密")
    print("方案B: 特征提取+特征加密（解耦方案）")

    # 定义测试分辨率
    resolutions = [(224, 224), (512, 512), (700, 460), (1024, 1024)]
    num_trials = 30

    print(f"\n测试分辨率: {resolutions}")
    print(f"每个配置试验次数: {num_trials}")

    # 运行实验
    print("\n开始实验...")
    results_df = run_experiment(resolutions, num_trials)

    # 显示结果
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    display_columns = ['分辨率', '方案', '平均时间_秒', '时间置信区间', '平均通信大小_字节', '通信大小置信区间']
    print(results_df[display_columns].round(4).to_string(index=False))

    # 统计分析
    statistical_analysis(results_df)

    # 绘制图表
    print("\n生成图表...")
    plot_results(results_df)

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"实验一_同态加密方案对比_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {results_filename}")

    print("\n实验完成！")


if __name__ == "__main__":
    main()