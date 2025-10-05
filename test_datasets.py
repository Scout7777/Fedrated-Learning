import torch
from datasets import load_dataset

# 加载数据集
BreakHis = load_dataset("chenqitao/BreakHis", split="train")

# 打印数据集基本信息
print("=== 数据集基本信息 ===")
print(f"数据集长度: {len(BreakHis)}")
print(f"特征列名: {BreakHis.column_names}")
print(f"数据集特征结构: {BreakHis.features}")

# 查看第一个样本
print("\n=== 第一个样本信息 ===")
first_sample = BreakHis[0]
print(f"样本键值: {first_sample.keys()}")

# 检查图像信息
for i in range(10):
    if 'image' in first_sample:
        img = first_sample['image']
        print(f"图像类型: {type(img)}")
        print(f"图像模式: {img.mode}")
        print(f"图像尺寸: {img.size}")


# 检查标签信息
if 'label' in first_sample:
    print(f"标签: {first_sample['label']}")
    # 查看所有可能的标签
    if hasattr(BreakHis.features['label'], 'names'):
        print(f"所有标签类别: {BreakHis.features['label'].names}")
        print(f"类别数量: {len(BreakHis.features['label'].names)}")




cifar10 = load_dataset("uoft-cs/cifar10", split="test")

# 打印数据集基本信息
print("=== 数据集基本信息 ===")
print(f"数据集长度: {len(cifar10)}")
print(f"特征列名: {cifar10.column_names}")
print(f"数据集特征结构: {cifar10.features}")

# 查看第一个样本
print("\n=== 第一个样本信息 ===")
first_sample = cifar10[0]
print(f"样本键值: {first_sample.keys()}")

# 检查图像信息
for i in range(10):
    if 'img' in first_sample:
        img = first_sample['img']
        print(f"图像类型: {type(img)}")
        print(f"图像模式: {img.mode}")
        print(f"图像尺寸: {img.size}")


# 检查标签信息
if 'label' in first_sample:
    print(f"标签: {first_sample['label']}")
    # 查看所有可能的标签
    if hasattr(BreakHis.features['label'], 'names'):
        print(f"所有标签类别: {BreakHis.features['label'].names}")
        print(f"类别数量: {len(BreakHis.features['label'].names)}")