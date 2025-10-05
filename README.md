# 同态加密的分布式协作医学影像异常检测
## 安装运行流程
### 1.安装配置环境
在anaconda创建一个叫fl0的虚拟环境，命令如下
```bash
conda create -n fl0 python==3.10
```
在虚拟环境中运行如下命令，安装flower联邦学习框架，所需的pytorch之类的库，tenseal同态加密库
```bash
pip install -U "flwr[simulation]"
cd flower-tutorial
pip install -e .
pip install tenseal
```
由于我们是在windows下运行该项目，可能有其他报错，请根据命令行错误提示安装一些东西

通过pycharm或者vscode配置你的python interpreter
### 2.训练模型并保存在服务器
在虚拟环境的terminal中输入如下命令
```bash
flwr run .
```
训练好的模型会保存为final_model.pth保存在项目根目录下
### 3.上传图片到服务器进行检测
直接运行fl1下的run即可，会上传根目录下的test图片


## 相关参考
flower官网


tenseal官网


