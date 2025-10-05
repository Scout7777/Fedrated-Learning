import base64
import json

from fl1.client import MedicalClient
from fl1.server import StandaloneMedicalServer


def main():
    # 客户端处理
    image_path = "C:/Users/houyuxuan/fl0/test.jpg"  # 替换为实际图像路径

    # 初始化客户端
    client = MedicalClient("C:/Users/houyuxuan/fl0/final_model.pt")

    # 预处理和加密
    serialized_data = client.client_preprocess(image_path)

    # 准备传输数据（包含加密上下文）
    request_data = {
        'encrypted_image': serialized_data,
        'context': base64.b64encode(client.context.serialize()).decode('utf-8')
    }

    # 模拟网络传输到服务器
    print("传输数据到服务器...")

    # 服务器端处理
    server = StandaloneMedicalServer("C:/Users/houyuxuan/fl0/final_model.pt")  # 替换为实际模型路径
    response = server.process_request(json.dumps(request_data))

    # 解析服务器响应
    response_data = json.loads(response)
    if response_data['status'] == 'success':
        # 客户端解密结果
        encrypted_result = client.deserialize_result(response_data['encrypted_result'])
        probabilities = client.decrypt_result(encrypted_result)

        print(f"最终预测结果: {probabilities}")
        print(f"诊断结果: {'阳性' if probabilities[0] > 0.5 else '阴性'}")
    else:
        print(f"处理失败: {response_data['message']}")


if __name__ == "__main__":
    main()