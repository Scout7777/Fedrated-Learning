import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time


# 简单的医疗图像分类网络（与您原有结构匹配）
class SimpleMedicalNet(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class HomomorphicEncryptionTester:
    def __init__(self, model_path=None, input_dim=64):
        # 创建同态加密上下文（与您提供的结构一致）
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[50, 30, 30, 30, 50]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** 30
        self.context.generate_relin_keys()  # 添加重线性化密钥

        # 加载或创建模型
        if model_path:
            self.model = SimpleMedicalNet(input_dim=input_dim)
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
        else:
            # 创建随机初始化的模型用于测试
            self.model = SimpleMedicalNet(input_dim=input_dim)
            torch.manual_seed(42)
            for p in self.model.parameters():
                nn.init.normal_(p, mean=0.0, std=0.1)

        self.model.eval()

    def encrypted_forward(self, encrypted_image):
        """同态加密前向传播（与您提供的结构一致）"""
        with torch.no_grad():
            # 提取权重和偏置，注意转置以匹配xW格式
            fc1_weight = self.model.fc1.weight.data.numpy().T
            fc1_bias = self.model.fc1.bias.data.numpy()
            fc2_weight = self.model.fc2.weight.data.numpy().T
            fc2_bias = self.model.fc2.bias.data.numpy()
            fc3_weight = self.model.fc3.weight.data.numpy().T
            fc3_bias = self.model.fc3.bias.data.numpy()

        # 同态加密计算
        result = encrypted_image.mm(fc1_weight) + fc1_bias
        result = result.mm(fc2_weight) + fc2_bias
        result = result.mm(fc3_weight) + fc3_bias
        return result

    def plain_forward(self, image):
        """明文前向传播"""
        with torch.no_grad():
            image_tensor = torch.from_numpy(image.astype(np.float32))
            output = self.model(image_tensor).numpy()
        return output

    def test_accuracy(self, test_features, test_labels, num_samples=100):
        """测试同态加密与明文推理的准确率"""
        results = []

        for i in range(min(num_samples, len(test_features))):
            # 获取测试样本
            feature = test_features[i].reshape(1, -1)
            true_label = test_labels[i]

            # 明文推理
            start_time = time.time()
            plain_output = self.plain_forward(feature)
            plain_time = time.time() - start_time
            plain_pred = np.argmax(plain_output)
            plain_correct = (plain_pred == true_label)

            # 同态加密推理
            start_time = time.time()
            try:
                # 加密输入
                encrypted_input = ts.ckks_vector(self.context, feature.flatten().tolist())
                # 同态计算
                encrypted_output = self.encrypted_forward(encrypted_input)
                # 解密结果
                decrypted_output = np.array(encrypted_output.decrypt()).reshape(1, -1)
                encrypted_time = time.time() - start_time

                encrypted_pred = np.argmax(decrypted_output)
                encrypted_correct = (encrypted_pred == true_label)

                # 计算误差
                abs_error = np.max(np.abs(plain_output - decrypted_output))
                rel_error = np.max(np.abs((plain_output - decrypted_output) / (np.abs(plain_output) + 1e-8)))

                success = True
                error_msg = ""

            except Exception as e:
                encrypted_time = 0
                encrypted_pred = -1
                encrypted_correct = False
                abs_error = float('inf')
                rel_error = float('inf')
                success = False
                error_msg = str(e)

            # 记录结果
            result = {
                '样本ID': i,
                '真实标签': true_label,
                '明文预测': plain_pred,
                '明文正确': plain_correct,
                '明文推理时间_秒': plain_time,
                '加密预测': encrypted_pred,
                '加密正确': encrypted_correct,
                '加密推理时间_秒': encrypted_time,
                '最大绝对误差': abs_error,
                '最大相对误差': rel_error,
                '测试成功': success,
                '错误信息': error_msg
            }
            results.append(result)

            if i % 10 == 0:
                print(f"测试进度: {i + 1}/{min(num_samples, len(test_features))}")

        return results


def create_synthetic_dataset(num_samples=1000, input_dim=64, num_classes=2):
    """创建合成测试数据集"""
    np.random.seed(42)

    # 生成特征
    features = np.random.randn(num_samples, input_dim).astype(np.float32) * 0.5

    # 生成标签（简单线性可分）
    weights = np.random.randn(input_dim, num_classes)
    logits = features @ weights
    labels = np.argmax(logits, axis=1)

    return features, labels


def main():
    # 设置中文字体（避免中文显示问题）
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 参数设置
    input_dim = 64
    num_classes = 2
    num_test_samples = 50  # 测试样本数量

    print("=== 同态加密准确率测试 ===")

    # 创建测试数据
    print("生成测试数据...")
    test_features, test_labels = create_synthetic_dataset(
        num_samples=200,
        input_dim=input_dim,
        num_classes=num_classes
    )

    # 初始化测试器
    print("初始化同态加密测试器...")
    tester = HomomorphicEncryptionTester(input_dim=input_dim)

    # 运行测试
    print(f"开始测试 {num_test_samples} 个样本...")
    results = tester.test_accuracy(test_features, test_labels, num_test_samples)

    # 分析结果
    df = pd.DataFrame(results)

    # 计算统计信息
    successful_tests = df[df['测试成功'] == True]
    total_tests = len(df)
    successful_count = len(successful_tests)

    if successful_count > 0:
        plain_accuracy = successful_tests['明文正确'].mean() * 100
        encrypted_accuracy = successful_tests['加密正确'].mean() * 100
        avg_abs_error = successful_tests['最大绝对误差'].mean()
        avg_rel_error = successful_tests['最大相对误差'].mean()
        avg_plain_time = successful_tests['明文推理时间_秒'].mean() * 1000  # 毫秒
        avg_encrypted_time = successful_tests['加密推理时间_秒'].mean() * 1000  # 毫秒
    else:
        plain_accuracy = encrypted_accuracy = avg_abs_error = avg_rel_error = 0
        avg_plain_time = avg_encrypted_time = 0

    # 打印详细结果表格
    print("\n" + "=" * 80)
    print("详细测试结果（前10个样本）:")
    print("=" * 80)

    # 显示前10个样本的详细结果
    display_columns = ['样本ID', '真实标签', '明文预测', '加密预测', '明文正确', '加密正确', '最大绝对误差']
    display_df = df[display_columns].head(10)
    print(display_df.to_string(index=False))

    # 打印统计总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    print(f"总测试样本数: {total_tests}")
    print(f"成功测试数: {successful_count} ({successful_count / total_tests * 100:.1f}%)")
    print(f"明文准确率: {plain_accuracy:.2f}%")
    print(f"加密准确率: {encrypted_accuracy:.2f}%")
    print(f"平均绝对误差: {avg_abs_error:.6f}")
    print(f"平均相对误差: {avg_rel_error:.6f}")
    print(f"平均明文推理时间: {avg_plain_time:.2f} 毫秒")
    print(f"平均加密推理时间: {avg_encrypted_time:.2f} 毫秒")
    print(f"时间开销比: {avg_encrypted_time / avg_plain_time if avg_plain_time > 0 else 'N/A':.1f} 倍")

    # 保存详细结果到CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"同态加密测试详细结果_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {filename}")

    # 保存统计总结到另一个CSV
    summary_data = {
        '总测试样本数': total_tests,
        '成功测试数': successful_count,
        '成功率': successful_count / total_tests,
        '明文准确率': plain_accuracy,
        '加密准确率': encrypted_accuracy,
        '平均绝对误差': avg_abs_error,
        '平均相对误差': avg_rel_error,
        '平均明文推理时间_毫秒': avg_plain_time,
        '平均加密推理时间_毫秒': avg_encrypted_time,
        '时间开销比': avg_encrypted_time / avg_plain_time if avg_plain_time > 0 else 0,
        '测试时间戳': timestamp
    }

    summary_df = pd.DataFrame([summary_data])
    summary_filename = f"同态加密测试总结_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
    print(f"测试总结已保存到: {summary_filename}")

    # 打印性能对比表格
    print("\n" + "=" * 50)
    print("性能对比:")
    print("=" * 50)
    performance_data = {
        '指标': ['准确率', '平均推理时间', '最大绝对误差'],
        '明文推理': [f"{plain_accuracy:.2f}%", f"{avg_plain_time:.2f} 毫秒", f"{avg_abs_error:.6f}"],
        '加密推理': [f"{encrypted_accuracy:.2f}%", f"{avg_encrypted_time:.2f} 毫秒", f"{avg_abs_error:.6f}"],
        '差异': [f"{encrypted_accuracy - plain_accuracy:+.2f}%",
                 f"{avg_encrypted_time / avg_plain_time if avg_plain_time > 0 else 'N/A':.1f}倍",
                 f"{avg_abs_error:.6f}"]
    }
    performance_df = pd.DataFrame(performance_data)
    print(performance_df.to_string(index=False))

    # 打印错误统计（如果有）
    failed_tests = df[df['测试成功'] == False]
    if len(failed_tests) > 0:
        print(f"\n失败测试统计 ({len(failed_tests)} 个):")
        error_counts = failed_tests['错误信息'].value_counts()
        for error, count in error_counts.items():
            print(f"  {error}: {count} 次")

    # 打印成功样本的误差分布
    if successful_count > 0:
        print(f"\n成功样本误差分布:")
        error_stats = successful_tests['最大绝对误差'].describe()
        print(f"  最小值: {error_stats['min']:.6f}")
        print(f"  最大值: {error_stats['max']:.6f}")
        print(f"  平均值: {error_stats['mean']:.6f}")
        print(f"  标准差: {error_stats['std']:.6f}")


if __name__ == "__main__":
    main()