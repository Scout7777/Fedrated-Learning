import numpy as np
import tenseal as ts
import torch
import torch.nn as nn
import base64
import json

from tenseal import CKKSVector

from fl0.net import SimpleMedicalNet


class MedicalServer:
    def __init__(self, model_path=None):
        self.model = SimpleMedicalNet()
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("模型加载成功")

    def process_encrypted_image(self, serialized_data, client_context):
        encrypted_image = self.deserialize_encrypted_data(serialized_data, client_context)
        encrypted_result = self.encrypted_forward(encrypted_image)
        result_data = self.serialize_encrypted_result(encrypted_result)
        return result_data

    def deserialize_encrypted_data(self, base64_data, context):
        serialized_data = base64.b64decode(base64_data.encode('utf-8'))
        encrypted_image = ts.ckks_vector_from(context, serialized_data)
        return encrypted_image

    def encrypted_forward(self, encrypted_image):
        with torch.no_grad():
            fc1_weight = self.model.fc1.weight.data.numpy().T  # 转置以匹配xW格式
            fc1_bias = self.model.fc1.bias.data.numpy()
            fc2_weight = self.model.fc2.weight.data.numpy().T
            fc2_bias = self.model.fc2.bias.data.numpy()
            fc3_weight = self.model.fc3.weight.data.numpy().T
            fc3_bias = self.model.fc3.bias.data.numpy()
        result = encrypted_image.mm(fc1_weight) + fc1_bias
        result = result.mm(fc2_weight) + fc2_bias

        result = result.mm(fc3_weight) + fc3_bias
        return result


    def serialize_encrypted_result(self, encrypted_result):
        """序列化加密结果"""
        serialized_data = encrypted_result.serialize()
        base64_data = base64.b64encode(serialized_data).decode('utf-8')
        return base64_data

# 服务器独立运行版本
class StandaloneMedicalServer(MedicalServer):
    def __init__(self, model_path):
        super().__init__(model_path)

    def process_request(self, client_data):
        """处理客户端请求"""
        # try:
            # 解析客户端数据
        data = json.loads(client_data)
        encrypted_data = data['encrypted_image']
        context_data = data['context']
        # 重建加密上下文
        context = ts.context_from(base64.b64decode(context_data.encode('utf-8')))
        # 处理加密图像
        result = self.process_encrypted_image(encrypted_data, context)
        response = {
            'status': 'success',
            'encrypted_result': result
        }
        return json.dumps(response)