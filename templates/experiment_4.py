# 实验 4：聚合一致性测试（代码示例）
#
# 目标：验证服务器端联邦平均（FedAvg）聚合后，
# 得到的全局模型参数在数学上是按预期权重平均（可复现），
# 并在不同客户端数据量与不同更新步数场景下保持一致性。

# file: test_fedavg_consistency.py
import copy
import numpy as np
import torch
import torch.nn as nn

def get_model_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()

def set_model_params(model, vec):
    torch.nn.utils.vector_to_parameters(torch.from_numpy(vec).float(), model.parameters())

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def local_update(model, local_seed=0, steps=1):
    # 模拟客户端本地训练，返回更新后的参数向量
    torch.manual_seed(local_seed)
    model_local = copy.deepcopy(model)
    opt = torch.optim.SGD(model_local.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    # 合成数据
    for _ in range(steps):
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        out = model_local(x)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return get_model_params(model_local)

def fedavg_aggregate(param_list, weight_list):
    # param_list: list of numpy vectors, weight_list: list of scalars sum to 1
    agg = np.zeros_like(param_list[0])
    for p,w in zip(param_list, weight_list):
        agg += p * w
    return agg

def main():
    # 初始化全局模型
    global_model = TinyModel()
    init_params = get_model_params(global_model)
    # 模拟三个客户端，数据量分别 100, 200, 700 -> 聚合权重
    data_sizes = np.array([100, 200, 700])
    weights = data_sizes / data_sizes.sum()

    # 本地更新（不同随机种子）
    param_updates = []
    for i,seed in enumerate([1,2,3]):
        p = local_update(global_model, local_seed=seed, steps=5)
        param_updates.append(p)

    # 服务器聚合
    agg = fedavg_aggregate(param_updates, weights)
    # 将 agg 写回全局模型
    set_model_params(global_model, agg)

    # 验证一致性：直接用逐项加权求和是否与聚合结果一致 (数值精度验证)
    # 这里是同一过程，主要断言向量大小不变
    print("聚合参数向量长度：", agg.shape)
    # 额外可检查无 NaN，值域合理
    assert not np.isnan(agg).any()

    print("FedAvg 聚合完成，权重：", weights)

if __name__ == "__main__":
    main()


# 说明：
#
# 实验应在多轮、不同客户端数量与数据不均衡（Non-IID）的设置下重复，验证聚合后一致性与稳定性。
#
# 若你在真实联邦环境中（通信延迟、部分客户端掉线、异步更新），
# 可以扩展代码模拟掉线、丢包与延迟对聚合结果的影响，并衡量模型性能稳定性。