import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import font_manager
import os
import matplotlib


def plot_dp_comparison(results):
    """绘制差分隐私对比结果"""
    plt.style.use('seaborn-v0_8')
    plt.rc("font", family="YouYuan")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 数据准备
    methods = ['无DP', '低噪声DP', '高噪声DP']
    accuracies = [np.mean(results['no_dp']),
                  np.mean(results['dp_low']),
                  np.mean(results['dp_high'])]
    stds = [np.std(results['no_dp']),
            np.std(results['dp_low']),
            np.std(results['dp_high'])]

    colors = ['#2E8B57', '#FFA500', '#FF6347']

    # 1. 准确率对比柱状图
    bars = ax1.bar(methods, accuracies, yerr=stds, capsize=5,
                   color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_title('差分隐私对模型准确率的影响', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 在柱子上添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. 准确率下降百分比
    base_acc = np.mean(results['no_dp'])
    accuracy_drops = [
        0,
        ((base_acc - np.mean(results['dp_low'])) / base_acc * 100),
        ((base_acc - np.mean(results['dp_high'])) / base_acc * 100)
    ]

    ax2.bar(methods, accuracy_drops, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('准确率下降百分比 (%)', fontsize=12)
    ax2.set_title('差分隐私导致的准确率下降', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 添加百分比标签
    for i, drop in enumerate(accuracy_drops):
        ax2.text(i, drop + 0.5, f'{drop:.1f}%',
                 ha='center', va='bottom', fontweight='bold')

    # 3. 多次实验的箱线图
    data_to_plot = [results['no_dp'], results['dp_low'], results['dp_high']]
    box_plot = ax3.boxplot(data_to_plot, labels=methods, patch_artist=True)

    # 设置箱线图颜色
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_ylabel('准确率', fontsize=12)
    ax3.set_title('多次实验准确率分布（箱线图）', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. 折线图显示趋势
    trials = range(1, len(results['no_dp']) + 1)
    ax4.plot(trials, results['no_dp'], 'o-', linewidth=2, markersize=8,
             label='无DP', color=colors[0])
    ax4.plot(trials, results['dp_low'], 's-', linewidth=2, markersize=8,
             label='低噪声DP', color=colors[1])
    ax4.plot(trials, results['dp_high'], '^-', linewidth=2, markersize=8,
             label='高噪声DP', color=colors[2])

    ax4.set_xlabel('实验次数', fontsize=12)
    ax4.set_ylabel('准确率', fontsize=12)
    ax4.set_title('多次实验准确率变化趋势', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(trials)

    plt.tight_layout()
    plt.savefig('dp_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_detailed_analysis(results):
    """绘制详细分析图表"""
    plt.rc("font", family="YouYuan")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 准备数据
    methods = ['无DP', '低噪声DP', '高噪声DP']
    mean_acc = [np.mean(results['no_dp']), np.mean(results['dp_low']), np.mean(results['dp_high'])]
    std_acc = [np.std(results['no_dp']), np.std(results['dp_low']), np.std(results['dp_high'])]

    # 1. 准确率分布小提琴图
    data_for_violin = []
    labels_for_violin = []
    for method, acc_list in zip(methods, [results['no_dp'], results['dp_low'], results['dp_high']]):
        data_for_violin.extend(acc_list)
        labels_for_violin.extend([method] * len(acc_list))

    df_violin = pd.DataFrame({'Accuracy': data_for_violin, 'Method': labels_for_violin})

    sns.violinplot(data=df_violin, x='Method', y='Accuracy', ax=ax1,
                   palette=['#2E8B57', '#FFA500', '#FF6347'])
    ax1.set_title('准确率分布小提琴图', fontsize=14, fontweight='bold')
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. 准确率变化热力图
    heatmap_data = np.array([results['no_dp'], results['dp_low'], results['dp_high']])
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(results['no_dp'])))
    ax2.set_xticklabels([f'试验{i + 1}' for i in range(len(results['no_dp']))])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(methods)
    ax2.set_title('准确率热力图', fontsize=14, fontweight='bold')

    # 添加数值标签
    for i in range(3):
        for j in range(len(results['no_dp'])):
            ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                     ha='center', va='center', fontweight='bold', fontsize=10)

    plt.colorbar(im, ax=ax2)

    # 3. 累积准确率
    cumulative_no_dp = np.cumsum(results['no_dp']) / np.arange(1, len(results['no_dp']) + 1)
    cumulative_dp_low = np.cumsum(results['dp_low']) / np.arange(1, len(results['dp_low']) + 1)
    cumulative_dp_high = np.cumsum(results['dp_high']) / np.arange(1, len(results['dp_high']) + 1)

    ax3.plot(range(1, len(results['no_dp']) + 1), cumulative_no_dp, 'o-',
             label='无DP', color='#2E8B57', linewidth=2)
    ax3.plot(range(1, len(results['dp_low']) + 1), cumulative_dp_low, 's-',
             label='低噪声DP', color='#FFA500', linewidth=2)
    ax3.plot(range(1, len(results['dp_high']) + 1), cumulative_dp_high, '^-',
             label='高噪声DP', color='#FF6347', linewidth=2)

    ax3.set_xlabel('实验次数', fontsize=12)
    ax3.set_ylabel('累积平均准确率', fontsize=12)
    ax3.set_title('累积平均准确率变化', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, len(results['no_dp']) + 1))

    # 4. 性能损失与隐私保护权衡
    privacy_levels = [0, 1, 2]  # 0: 无DP, 1: 低噪声, 2: 高噪声
    performance_loss = [0,
                        (mean_acc[0] - mean_acc[1]) / mean_acc[0] * 100,
                        (mean_acc[0] - mean_acc[2]) / mean_acc[0] * 100]

    ax4.plot(privacy_levels, performance_loss, 'o--', linewidth=3, markersize=10,
             color='purple', markerfacecolor='red')
    ax4.set_xlabel('隐私保护级别', fontsize=12)
    ax4.set_ylabel('性能损失 (%)', fontsize=12)
    ax4.set_title('隐私保护 vs 性能损失权衡', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(privacy_levels)
    ax4.set_xticklabels(['无保护', '低保护', '高保护'])

    # 添加数据点标签
    for i, (x, y) in enumerate(zip(privacy_levels, performance_loss)):
        ax4.annotate(f'{y:.1f}%', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('dp_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_table(results):
    """创建结果汇总表格"""
    methods = ['无DP', '低噪声DP', '高噪声DP']
    data_lists = [results['no_dp'], results['dp_low'], results['dp_high']]

    summary_data = []
    for method, data in zip(methods, data_lists):
        summary_data.append({
            '方法': method,
            '平均准确率': f'{np.mean(data):.4f}',
            '标准差': f'{np.std(data):.4f}',
            '最低准确率': f'{np.min(data):.4f}',
            '最高准确率': f'{np.max(data):.4f}',
            '准确率下降(%)': f'{((np.mean(results["no_dp"]) - np.mean(data)) / np.mean(results["no_dp"]) * 100):.2f}'
        })

    df = pd.DataFrame(summary_data)
    return df


def main_plotting(results):
    """主绘图函数"""
    print("生成差分隐私对比实验图表...")

    # 创建汇总表格
    summary_df = create_summary_table(results)
    print("\n=== 实验结果汇总 ===")
    print(summary_df.to_string(index=False))

    # 绘制主要对比图表
    plot_dp_comparison(results)

    # 绘制详细分析图表
    plot_detailed_analysis(results)

    # 保存结果到文件
    summary_df.to_csv('dp_experiment_results.csv', index=False, encoding='utf-8-sig')
    print("\n结果已保存到: dp_experiment_results.csv")
    print("图表已保存为: dp_comparison_results.png 和 dp_detailed_analysis.png")


# 如果直接运行此文件，可以使用示例数据测试
if __name__ == "__main__":

    # 示例数据
    example_results = {
        'no_dp': [0.85, 0.83, 0.86, 0.84, 0.87],
        'dp_low': [0.78, 0.76, 0.79, 0.77, 0.78],
        'dp_high': [0.65, 0.63, 0.66, 0.64, 0.65]
    }

    main_plotting(example_results)