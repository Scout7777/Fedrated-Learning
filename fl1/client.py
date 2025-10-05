import tenseal as ts
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
from fl1.server import MedicalServer

class SimpleMedicalNet(nn.Module):
    """简单的医学影像二分类网络"""

    def __init__(self):
        super(SimpleMedicalNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_input_size = 16
        self.fc1 = nn.Linear(self.fc_input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.global_avg_pool(x)  # 输出形状: [batch, 16, 1, 1]
        x = x.view(-1, self.fc_input_size)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class MedicalClient:
    def __init__(self, model_path=None):
        # 创建同态加密上下文
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[50, 30, 30, 30, 50]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** 30

        # 加载模型用于客户端预处理
        if model_path:
            self.model = SimpleMedicalNet()
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def extract_features(self, image_tensor):
        """在客户端提取特征（卷积+池化），只加密全连接层的输入"""
        with torch.no_grad():
            # 进行卷积和池化操作
            x = self.model.pool(self.model.conv1(image_tensor))
            x = self.model.pool(self.model.conv2(x))
            x = self.model.global_avg_pool(x)
            # 展平特征
            features = x.view(-1, self.model.fc_input_size)
            return features

    def encrypt_features(self, features_tensor):
        """加密特征向量"""
        features_data = features_tensor.numpy().flatten()
        encrypted_features = ts.ckks_vector(self.context, features_data)
        return encrypted_features

    def client_preprocess(self, image_path):
        """客户端完整预处理流程"""
        # 1. 预处理图像
        image_tensor = self.preprocess_image(image_path)

        # 2. 提取特征（卷积+池化）
        features = self.extract_features(image_tensor)

        print("特征尺寸：")
        print(features.shape)

        # 3. 加密特征
        encrypted_features = self.encrypt_features(features)
        ciphertexts = encrypted_features.ciphertext()
        if len(ciphertexts) > 1:
            print(f"警告: 加密特征被分块为 {len(ciphertexts)} 个部分")

        # 4. 序列化
        serialized_data = self.serialize_encrypted_data(encrypted_features)

        return serialized_data

    def preprocess_image(self, image_path):
        """预处理图像为模型输入格式"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((700, 460))

        # 转换为tensor并归一化
        image_tensor = torch.from_numpy(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # CHW格式
        image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度

        return image_tensor

    def encrypt_image(self, image_tensor):
        """加密图像数据"""
        # 将tensor转换为numpy数组并展平
        image_data = image_tensor.numpy().flatten()

        # 使用CKKS加密
        encrypted_image = ts.ckks_vector(self.context, image_data)

        return encrypted_image

    def serialize_encrypted_data(self, encrypted_image):
        """序列化加密数据"""
        serialized_data = encrypted_image.serialize()
        # 转换为base64便于传输
        base64_data = base64.b64encode(serialized_data).decode('utf-8')
        return base64_data

    def deserialize_result(self, base64_data):
        """反序列化服务器返回的加密结果"""
        serialized_data = base64.b64decode(base64_data.encode('utf-8'))
        encrypted_result = ts.ckks_vector_from(self.context, serialized_data)
        return encrypted_result

    def decrypt_result(self, encrypted_result):
        """解密结果"""
        decrypted_result = encrypted_result.decrypt()
        result_tensor = torch.tensor(decrypted_result)

        # 应用sigmoid得到概率
        probabilities = torch.sigmoid(result_tensor)

        return probabilities




# 客户端使用示例
def client_process(image_path):
    client = MedicalClient()

    # 1. 预处理图像
    print("预处理图像...")
    image_tensor = client.preprocess_image(image_path)

    # 2. 加密图像
    print("加密图像数据...")
    encrypted_image = client.encrypt_image(image_tensor)



    # 3. 序列化并准备传输
    print("序列化加密数据...")
    serialized_data = client.serialize_encrypted_data(encrypted_image)

    # 这里模拟网络传输，实际中会发送HTTP请求
    print("发送加密数据到服务器...")

    # 模拟服务器处理并返回结果
    server_result = simulate_server_processing(serialized_data, client.context)

    # 4. 接收并反序列化结果
    print("接收服务器结果...")
    encrypted_result = client.deserialize_result(server_result)

    # 5. 解密结果
    print("解密结果...")
    probabilities = client.decrypt_result(encrypted_result)

    print(f"预测概率: {probabilities}")
    print(f"预测类别: {'阳性' if probabilities[0] > 0.5 else '阴性'}")

    return probabilities


def simulate_server_processing(serialized_data, context):
    """模拟服务器处理过程"""
    # 在实际应用中，这部分会在服务器端运行
    server = MedicalServer()
    return server.process_encrypted_image(serialized_data, context)