"""fl0: A Flower / pytorch_msg_api app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy import FedAdagrad
from torch.backends.opt_einsum import strategy
from fl0.task import central_evaluate

from fl0.net import SimpleMedicalNet

# Create ServerApp
app = ServerApp()

#python装饰器，把之后的main函数传递给app.main（），用该装饰器返回的函数替代main函数
@app.main()
#类型注解并返回none
#grid是联邦学习中的通信节点，是客户端到服务器的中间节点 context是训练参数

def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    #训练设置在toml里
    fraction_train: float = context.run_config["fraction-train"]#0.5
    num_rounds: int = context.run_config["num-server-rounds"]#3
    lr: float = context.run_config["lr"]#0.01

    # Load global model
    global_model = SimpleMedicalNet()
    #把nn的参数转为flower可识别的参数
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    # strategy = FedAvg(fraction_train=fraction_train)
    strategy = FedAdagrad(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
