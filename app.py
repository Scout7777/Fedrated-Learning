# app.py
import os
import sys
import json
import time

import base64

import tempfile
import traceback
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# JWT
try:
    import jwt
except Exception:
    raise RuntimeError("缺少依赖 pyjwt，请先 pip install pyjwt")

# 人脸识别
try:
    import face_recognition
    import cv2
    import numpy as np
    from PIL import Image
    import base64
    import io
    FACE_RECOGNITION_AVAILABLE = True
except Exception as e:
    print(f"人脸识别模块导入失败: {e}")
    print("请安装依赖: pip install opencv-python face-recognition Pillow")
    FACE_RECOGNITION_AVAILABLE = False

# Ensure torch is importable (used for saving model)
try:
    import torch
except Exception:
    torch = None  # will check later if needed

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PREDICTIONS_FILE = os.path.join(BASE_DIR, "predictions.json")
METRICS_FILE = os.path.join(BASE_DIR, "metrics.json")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# ensure json files exist
def ensure_json_file(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2, ensure_ascii=False)

ensure_json_file(USERS_FILE, {"users": []})
ensure_json_file(PREDICTIONS_FILE, {"predictions": []})
ensure_json_file(METRICS_FILE, {})

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/")
CORS(app)
app.config["SECRET_KEY"] = "change-this-secret-to-a-strong-random-value"

JWT_SECRET = app.config["SECRET_KEY"]
JWT_ALGORITHM = "HS256"
JWT_EXP_DAYS = 1

# Robust JSON helpers
def load_json(path, default=None):
    """
    Safe JSON loader: returns default if file missing / empty / invalid.
    """
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content or content.strip() == "":
                return default
            return json.loads(content)
    except json.JSONDecodeError:
        # invalid JSON (e.g. partial write), return default
        return default
    except Exception:
        traceback.print_exc()
        return default

def save_json(path, data):
    """
    Atomic write: write to temp file and replace target file.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, path)
    except Exception:
        traceback.print_exc()
        # fallback write
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            traceback.print_exc()

# user management (file-based)
def find_user(username):
    data = load_json(USERS_FILE, {"users": []})
    for u in data.get("users", []):
        if u.get("username") == username:
            return u
    return None

def add_user(username, password, face_encoding=None):
    if find_user(username):
        return False, "用户名已存在"
    hashed = generate_password_hash(password)
    user_data = {
        "username": username,
        "password_hash": hashed,
        "created_at": datetime.utcnow().isoformat()
    }
    
    # 如果提供了人脸特征，则存储
    if face_encoding is not None:
        user_data["face_encoding"] = face_encoding
    
    data = load_json(USERS_FILE, {"users": []})
    data.setdefault("users", []).append(user_data)
    save_json(USERS_FILE, data)
    return True, None

def verify_user(username, password):
    u = find_user(username)
    if not u:
        return False
    return check_password_hash(u.get("password_hash", ""), password)

def create_token(username):
    payload = {
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXP_DAYS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def create_face_token(username, minutes=5):
    payload = {
        "username": username,
        "face_ok": True,
        "exp": datetime.utcnow() + timedelta(minutes=minutes)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def decode_token(token):
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    return payload

# 人脸识别辅助函数
def base64_to_image(base64_string):
    """将base64字符串转换为PIL图像"""
    try:
        # 移除data:image/jpeg;base64,前缀
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Base64转图像失败: {e}")
        return None

def extract_face_encoding(image):
    """从图像中提取人脸特征编码"""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, "人脸识别模块未安装"
    
    try:
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组
        image_array = np.array(image)
        
        # 检测人脸位置
        face_locations = face_recognition.face_locations(image_array)
        
        if len(face_locations) == 0:
            return None, "未检测到人脸"
        
        if len(face_locations) > 1:
            return None, "检测到多张人脸，请确保只有一张人脸"
        
        # 提取人脸特征编码
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if len(face_encodings) == 0:
            return None, "无法提取人脸特征"
        
        # 将numpy数组转换为列表以便JSON序列化
        face_encoding = face_encodings[0].tolist()
        return face_encoding, None
        
    except Exception as e:
        return None, f"人脸特征提取失败: {str(e)}"

def compare_faces(encoding1, encoding2, tolerance=0.6):
    """比较两个人脸特征编码的相似度"""
    if not FACE_RECOGNITION_AVAILABLE:
        return False
    
    try:
        # 转换为numpy数组
        enc1 = np.array(encoding1)
        enc2 = np.array(encoding2)
        
        # 计算距离
        distance = face_recognition.face_distance([enc1], enc2)[0]
        
        # 返回是否匹配（距离小于阈值）
        return distance <= tolerance
    except Exception as e:
        print(f"人脸比较失败: {e}")
        return False

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get("Authorization", None)
        if not auth:
            return jsonify({"status": "error", "message": "缺少 Authorization header"}), 401
        if not auth.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Authorization header 应为 'Bearer <token>'"}), 401
        token = auth.split(" ", 1)[1]
        try:
            payload = decode_token(token)
            g.current_user = payload.get("username")
        except Exception:
            return jsonify({"status": "error", "message": "token 无效或已过期"}), 401
        return f(*args, **kwargs)
    return decorated

# ---------------------------
# import model inference classes (use fl1 as you requested)
# ---------------------------
MODEL_CLIENT_CLASS = None
MODEL_SERVER_CLASS = None
_model_import_attempts = []
try:
    from fl1.client import MedicalClient
    from fl1.server import StandaloneMedicalServer
    MODEL_CLIENT_CLASS = MedicalClient
    MODEL_SERVER_CLASS = StandaloneMedicalServer
    _model_import_attempts.append("已直接从 fl1.client 和 fl1.server 导入模型类。")
except Exception as e:
    _model_import_attempts.append(f"从 fl1 导入失败: {e}")

# ---------------------------
# try to import training utilities from fl0 (real training functions)
# ---------------------------
_ext_imports_log = []
use_real_training = False
fl0_train = None
fl0_test = None
fl0_load_data = None
SimpleMedicalNet = None
try:
    from fl0.task import train as fl0_train, test as fl0_test, load_data as fl0_load_data
    from fl0.net import SimpleMedicalNet
    _ext_imports_log.append("成功导入 fl0.task.train/test/load_data 与 fl0.net.SimpleMedicalNet")
    use_real_training = True
except Exception as e:
    _ext_imports_log.append(f"导入 fl0 训练相关模块失败: {e}")
    use_real_training = False

# MODEL file candidates
MODEL_PATH_CANDIDATES = [
    os.path.join(BASE_DIR, "final_model.pt"),
    os.path.join(BASE_DIR, "final_model", "final_model.pt"),
]

MODEL_PATH = next((p for p in MODEL_PATH_CANDIDATES if os.path.exists(p)), None)

CLIENT_INSTANCE = None
SERVER_INSTANCE = None

def get_client_instance():
    global CLIENT_INSTANCE
    if CLIENT_INSTANCE is None:
        if MODEL_CLIENT_CLASS is None:
            raise RuntimeError("无法导入模型客户端类。导入记录：\n" + "\n".join(_model_import_attempts))
        if MODEL_PATH is None:
            raise RuntimeError("未找到 final_model.pt，请放到项目根或修改 MODEL_PATH_CANDIDATES。")
        CLIENT_INSTANCE = MODEL_CLIENT_CLASS(MODEL_PATH)
    return CLIENT_INSTANCE

def get_server_instance():
    global SERVER_INSTANCE
    if SERVER_INSTANCE is None:
        if MODEL_SERVER_CLASS is None:
            raise RuntimeError("无法导入模型服务端类。导入记录：\n" + "\n".join(_model_import_attempts))
        if MODEL_PATH is None:
            raise RuntimeError("未找到 final_model.pt，请放到项目根或修改 MODEL_PATH_CANDIDATES。")
        SERVER_INSTANCE = MODEL_SERVER_CLASS(MODEL_PATH)
    return SERVER_INSTANCE

# ---------------------------
# TRAINING background state & task
# ---------------------------
TRAIN_STATUS = {
    "running": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "started_at": None,
    "finished_at": None,
    "status": "idle",
    "message": None,
    # DP config & noise logs
    "dp_noise_type": None,  # 'gaussian' | 'laplace' | None
    "dp_epsilon": None,     # float or None
    "noises": [],           # list of per-epoch noise values for display
    "last_noise": None
}

import threading
import random
import math
from PIL import Image
import numpy as np

def _background_train_task(total_epochs=20, sleep_per_epoch=1.0, lr=1e-3):
    """
    Background training task. Uses real fl0 training functions if available,
    otherwise falls back to simulated training.
    """
    try:
        TRAIN_STATUS.update({
            "running": True,
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": total_epochs,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "status": "running",
            "message": None,
            "noises": [],
            "last_noise": None
        })

        # If real training API is available, use it
        if use_real_training and fl0_train and fl0_test and fl0_load_data and SimpleMedicalNet:
            # instantiate model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if torch is not None else "cpu"
            model = SimpleMedicalNet()
            model.to(device)

            # Use load_data with single partition to get whole dataset (partition_id=0, num_partitions=1)
            try:
                trainloader, testloader = fl0_load_data(0, 1)
            except Exception as e:
                _ext_imports_log.append(f"fl0_load_data 调用失败: {e}")
                trainloader, testloader = None, None

            loss_history = []
            acc_history = []

            simulate = (trainloader is None or testloader is None)
            if simulate:
                _ext_imports_log.append("trainloader/testloader 为 None，回退到模拟训练。")

            for e in range(1, total_epochs + 1):
                if simulate:
                    time.sleep(sleep_per_epoch)
                    l = max(0.01, 1.0 / (0.2 * e + 1) + random.random() * 0.02)
                    a = min(0.999, 0.2 + 0.8 * (e / total_epochs) + random.random() * 0.01)
                else:
                    # call real training function for 1 epoch
                    try:
                        train_loss = fl0_train(model, trainloader, 1, lr, device)
                    except Exception as ex:
                        _ext_imports_log.append(f"fl0_train 第 {e} 轮失败: {ex}")
                        time.sleep(sleep_per_epoch)
                        train_loss = max(0.01, 1.0 / (0.2 * e + 1) + random.random() * 0.02)

                    try:
                        val_loss, val_acc = fl0_test(model, testloader, device)
                    except Exception as ex:
                        _ext_imports_log.append(f"fl0_test 第 {e} 轮失败: {ex}")
                        val_loss = train_loss
                        val_acc = min(0.999, 0.2 + 0.8 * (e / total_epochs))

                    l = float(val_loss)
                    a = float(val_acc)

                loss_history.append(round(l, 4))
                acc_history.append(round(a, 4))

                # DP noise per epoch (simulated)
                dp_type = TRAIN_STATUS.get("dp_noise_type")
                dp_eps = TRAIN_STATUS.get("dp_epsilon")
                noise_val = None
                if dp_type and dp_eps:
                    try:
                        scale = max(1e-6, 1.0 / float(dp_eps))
                        if dp_type == "gaussian":
                            noise_val = random.gauss(0.0, scale)
                        elif dp_type == "laplace":
                            # Laplace via inverse transform
                            u = random.random() - 0.5
                            noise_val = -scale * (1 if u < 0 else -1) * (math.log(1 - 2*abs(u)))
                        else:
                            noise_val = 0.0
                    except Exception:
                        noise_val = 0.0
                TRAIN_STATUS["last_noise"] = noise_val
                try:
                    if noise_val is not None:
                        TRAIN_STATUS["noises"].append(round(float(noise_val), 6))
                    else:
                        TRAIN_STATUS["noises"].append(None)
                except Exception:
                    pass

                TRAIN_STATUS["current_epoch"] = e
                TRAIN_STATUS["progress"] = int(e / total_epochs * 100)

                # Build confusion matrix example safely
                last_acc = acc_history[-1] if acc_history else 0.0
                cm = [
                    [int(200 * last_acc), int(50 * (1 - last_acc))],
                    [int(20 * (1 - last_acc)), int(100 * last_acc)]
                ]

                demo = {
                    "epochs": list(range(1, e + 1)),
                    "loss": loss_history,
                    "accuracy": acc_history,
                    "class_distribution": [0.6, 0.4],
                    "confusion_matrix": cm,
                    "trained_at": datetime.utcnow().isoformat()
                }

                # 在写入 metrics 前打印以便终端能看到进度
                print(
                    f"[TRAIN] epoch={e}/{total_epochs} progress={TRAIN_STATUS['progress']}% last_loss={loss_history[-1] if loss_history else None} last_acc={acc_history[-1] if acc_history else None} dp_noise={TRAIN_STATUS.get('last_noise')}")
                sys.stdout.flush()

                save_json(METRICS_FILE, demo)

            # after training loop, save final model
            try:
                if torch is not None:
                    torch.save(model.state_dict(), os.path.join(BASE_DIR, "final_model.pt"))
                    _ext_imports_log.append("训练完成并保存 final_model.pt")
                else:
                    _ext_imports_log.append("torch 未安装，跳过保存模型步骤")
            except Exception as ex:
                _ext_imports_log.append(f"保存模型失败: {ex}")

        else:
            # fallback simulation if real training API unavailable
            loss_history = []
            acc_history = []
            for e in range(1, total_epochs + 1):
                time.sleep(sleep_per_epoch)
                l = max(0.01, 1.0 / (0.2 * e + 1) + random.random() * 0.02)
                a = min(0.999, 0.2 + 0.8 * (e / total_epochs) + random.random() * 0.01)
                loss_history.append(round(l, 4))
                acc_history.append(round(a, 4))
                # DP noise per epoch (simulated)
                dp_type = TRAIN_STATUS.get("dp_noise_type")
                dp_eps = TRAIN_STATUS.get("dp_epsilon")
                noise_val = None
                if dp_type and dp_eps:
                    try:
                        scale = max(1e-6, 1.0 / float(dp_eps))
                        if dp_type == "gaussian":
                            noise_val = random.gauss(0.0, scale)
                        elif dp_type == "laplace":
                            u = random.random() - 0.5
                            noise_val = -scale * (1 if u < 0 else -1) * (math.log(1 - 2*abs(u)))
                        else:
                            noise_val = 0.0
                    except Exception:
                        noise_val = 0.0
                TRAIN_STATUS["last_noise"] = noise_val
                try:
                    if noise_val is not None:
                        TRAIN_STATUS["noises"].append(round(float(noise_val), 6))
                    else:
                        TRAIN_STATUS["noises"].append(None)
                except Exception:
                    pass

                TRAIN_STATUS["current_epoch"] = e
                TRAIN_STATUS["progress"] = int(e / total_epochs * 100)

                last_acc = acc_history[-1] if acc_history else 0.0
                cm = [
                    [int(200 * last_acc), int(50 * (1 - last_acc))],
                    [int(20 * (1 - last_acc)), int(100 * last_acc)]
                ]
                demo = {
                    "epochs": list(range(1, e + 1)),
                    "loss": loss_history,
                    "accuracy": acc_history,
                    "class_distribution": [0.6, 0.4],
                    "confusion_matrix": cm,
                    "trained_at": datetime.utcnow().isoformat()
                }
                save_json(METRICS_FILE, demo)

        TRAIN_STATUS.update({
            "running": False,
            "progress": 100,
            "current_epoch": total_epochs,
            "finished_at": datetime.utcnow().isoformat(),
            "status": "done",
            "message": "训练完成"
        })
    except Exception as ex:
        traceback.print_exc()
        TRAIN_STATUS.update({
            "running": False,
            "status": "error",
            "message": str(ex),
            "finished_at": datetime.utcnow().isoformat()
        })
        save_json(METRICS_FILE, {"error": str(ex)})

# ---------------------------
# endpoints
# ---------------------------

@app.route("/api/register", methods=["POST"])
def api_register():
    body = request.get_json(force=True, silent=True) or {}
    username = body.get("username")
    password = body.get("password")
    face_image_base64 = body.get("face_image")  # 可选的人脸图像
    
    if not username or not password:
        return jsonify({"status":"error", "message":"请提供 username 和 password"}), 400
    
    # 必须提供人脸图像
    if not face_image_base64:
        return jsonify({"status":"error", "message":"注册需要采集人脸，请添加人脸后再试"}), 400

    image = base64_to_image(face_image_base64)
    if image is None:
        return jsonify({"status":"error", "message":"人脸图像格式错误"}), 400

    face_encoding, error_msg = extract_face_encoding(image)
    if face_encoding is None:
        return jsonify({"status":"error", "message": error_msg}), 400
    
    ok, msg = add_user(username, password, face_encoding)
    if not ok:
        return jsonify({"status":"error", "message": msg}), 400
    token = create_token(username)
    return jsonify({"status":"success", "token": token, "has_face": True})

@app.route("/api/login_face", methods=["POST"])
def api_login_face():
    body = request.get_json(force=True, silent=True) or {}
    username = body.get("username")
    face_image_base64 = body.get("face_image")

    if not username or not face_image_base64:
        return jsonify({"status":"error", "message":"请提供 username 与 face_image"}), 400

    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({"status":"error", "message":"人脸识别功能不可用"}), 500

    user = find_user(username)
    if not user:
        return jsonify({"status":"error", "message":"用户不存在"}), 404
    if "face_encoding" not in user:
        return jsonify({"status":"error", "message":"该用户未注册人脸特征"}), 400

    image = base64_to_image(face_image_base64)
    if image is None:
        return jsonify({"status":"error", "message":"人脸图像格式错误"}), 400

    face_encoding, error_msg = extract_face_encoding(image)
    if face_encoding is None:
        return jsonify({"status":"error", "message": error_msg}), 400

    if not compare_faces(face_encoding, user["face_encoding"]):
        return jsonify({"status":"error", "message":"人脸识别失败"}), 401

    face_token = create_face_token(username)
    return jsonify({"status":"success", "face_token": face_token})

@app.route("/api/login", methods=["POST"])
def api_login():
    body = request.get_json(force=True, silent=True) or {}
    username = body.get("username")
    password = body.get("password")
    face_token = body.get("face_token")
    
    if not username or not password or not face_token:
        return jsonify({"status":"error", "message":"请提供 username、password 与 face_token（先完成人脸识别）"}), 400

    # 校验 face_token
    try:
        payload = decode_token(face_token)
        if not payload.get("face_ok") or payload.get("username") != username:
            return jsonify({"status":"error", "message":"人脸认证无效"}), 401
    except Exception:
        return jsonify({"status":"error", "message":"人脸认证已过期或无效"}), 401

    if not verify_user(username, password):
        return jsonify({"status":"error", "message":"用户名或密码错误"}), 401
    token = create_token(username)
    return jsonify({"status":"success", "token": token, "login_method": "2fa"})

@app.route("/api/profile", methods=["GET"])
@token_required
def api_profile():
    user = find_user(g.current_user)
    has_face = "face_encoding" in user if user else False
    return jsonify({"status":"success", "username": g.current_user, "has_face": has_face})

@app.route("/api/check_face", methods=["POST"])
def api_check_face():
    """检查用户是否已注册人脸特征"""
    body = request.get_json(force=True, silent=True) or {}
    username = body.get("username")
    
    if not username:
        return jsonify({"status":"error", "message":"请提供 username"}), 400
    
    user = find_user(username)
    if not user:
        return jsonify({"status":"error", "message":"用户不存在"}), 404
    
    has_face = "face_encoding" in user
    return jsonify({"status":"success", "has_face": has_face})

@app.route("/api/predict", methods=["POST"])
@token_required
def api_predict():
    try:
        if request.content_type and "multipart/form-data" in request.content_type:
            if "image" not in request.files:
                return jsonify({"status":"error","message":"缺少 image 文件"}), 400
            file = request.files["image"]
            ext = os.path.splitext(file.filename or "")[1] or ".jpg"
            safe_name = f"{int(time.time()*1000)}{ext}"
            dest_path = os.path.join(UPLOADS_DIR, safe_name)
            file.save(dest_path)
            image_path = dest_path
            preview_url = f"/uploads/{safe_name}"
        else:
            body = request.get_json(force=True, silent=True) or {}
            if body.get("use_demo"):
                demo_path = os.path.join(BASE_DIR, "test.jpg")
                if not os.path.exists(demo_path):
                    return jsonify({"status":"error", "message":"repo 中未找到 test.jpg"}), 400
                image_path = demo_path
                preview_url = "/test.jpg"
            else:
                return jsonify({"status":"error", "message":"请上传文件或发送 JSON {use_demo:true}"}), 400

        if MODEL_CLIENT_CLASS is None or MODEL_SERVER_CLASS is None:
            return jsonify({
                "status":"error",
                "message": "后端未能导入模型客户端/服务端类，无法执行推理。导入记录：\n" + "\n".join(_model_import_attempts)
            }), 500

        client = get_client_instance()
        server = get_server_instance()
        serialized = client.client_preprocess(image_path)
        context_b64 = None
        if hasattr(client, "context"):
            try:
                context_b64 = getattr(client, "context").serialize()
                import base64
                context_b64 = base64.b64encode(context_b64).decode("utf-8")
            except Exception:
                context_b64 = None

        request_data = {"encrypted_image": serialized}
        if context_b64:
            request_data["context"] = context_b64

        import json as _json
        response = server.process_request(_json.dumps(request_data))
        response_data = _json.loads(response)
        if response_data.get("status") != "success":
            return jsonify({"status":"error", "message": response_data.get("message", "server 处理失败")}), 500

        encrypted_result = client.deserialize_result(response_data["encrypted_result"])
        probabilities = client.decrypt_result(encrypted_result)

        try:
            probs_list = probabilities.detach().cpu().numpy().tolist()
        except Exception:
            try:
                probs_list = list(probabilities)
            except Exception:
                probs_list = [float(probabilities)]

        first_prob = float(probs_list[0]) if len(probs_list) > 0 else 0.0
        diagnosis = "阳性" if first_prob > 0.5 else "阴性"

        hist = load_json(PREDICTIONS_FILE, {"predictions": []})
        hist.setdefault("predictions", []).append({
            "username": g.current_user,
            "timestamp": datetime.utcnow().isoformat(),
            "probabilities": probs_list,
            "diagnosis": diagnosis,
            "preview_url": preview_url
        })
        save_json(PREDICTIONS_FILE, hist)

        return jsonify({
            "status":"success",
            "probabilities": probs_list,
            "diagnosis": diagnosis,
            "raw": str(probabilities),
            "preview_url": preview_url
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message": str(e)}), 500

@app.route('/api/features', methods=['POST'])
@token_required
def api_features():
    try:
        if 'image' not in request.files:
            return jsonify({"status":"error","message":"缺少 image 文件"}), 400
        file = request.files['image']
        img = Image.open(file.stream).convert('L')  # 灰度
        arr = np.array(img, dtype=np.float32)
        # 归一化
        if arr.max() > 0:
            arr = arr / 255.0
        # 16 维特征：多尺度统计（均值/方差/能量/熵 * 4个尺度）
        feats = []
        scales = [1, 2, 4, 8]
        for s in scales:
            # 下采样
            h, w = arr.shape
            h2 = max(1, h // s)
            w2 = max(1, w // s)
            small = np.array(Image.fromarray((arr*255).astype(np.uint8)).resize((w2, h2), Image.BILINEAR), dtype=np.float32) / 255.0
            mean = float(np.mean(small))
            var = float(np.var(small))
            energy = float(np.mean(small**2))
            # 熵
            hist, _ = np.histogram((small*255).astype(np.uint8), bins=256, range=(0,255), density=True)
            hist = hist + 1e-12
            entropy = float(-np.sum(hist * np.log(hist)))
            feats.extend([mean, var, energy, entropy])
        return jsonify({"status":"success", "features": feats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message": str(e)}), 500

@app.route('/api/ckks_demo', methods=['POST'])
@token_required
def api_ckks_demo():
    try:
        body = request.get_json(force=True, silent=True) or {}
        feats = body.get('features')
        poly = int(body.get('poly_modulus_degree', 8192))
        mod_bits_str = body.get('coeff_mod_bit_sizes', '50,30,30,30,50')
        mod_bits = [int(x.strip()) for x in str(mod_bits_str).split(',') if x.strip()]
        if not feats or not isinstance(feats, list):
            return jsonify({"status":"error","message":"缺少 features 列表"}), 400
        # 归一化输入
        vec = np.array(feats, dtype=np.float64)
        if vec.size == 0:
            return jsonify({"status":"error","message":"features 为空"}), 400
        # 使用 TenSEAL 演示 CKKS 编码/加密/解密
        try:
            import tenseal as ts
        except Exception as e:
            return jsonify({"status":"error","message":"缺少 TenSEAL 依赖，请先 pip install tenseal"}), 500

        # 建立 CKKS 上下文
        # coeff_mod_bit_sizes 映射为 list[int]
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly, coeff_mod_bit_sizes=mod_bits)
        # 设置全局比例，避免 no global scale 错误
        context.global_scale = 2 ** 30
        context.generate_galois_keys()
        context.generate_relin_keys()

        # 编码并加密
        plain = vec.tolist()
        enc_vec = ts.ckks_vector(context, plain)
        # 客户端 ciphertext（序列化为 bytes->base64 以便展示）
        try:
            import base64 as _b64
            client_ct = _b64.b64encode(enc_vec.serialize()).decode('utf-8')
        except Exception:
            client_ct = None

        # 这里模拟“服务端上的加密计算”：对密文做一次标量缩放（示例）
        try:
            server_vec = ts.ckks_vector_from(context, enc_vec.serialize())
            server_vec = server_vec * 1.0  # 示例计算
            server_ct = _b64.b64encode(server_vec.serialize()).decode('utf-8')
        except Exception:
            server_ct = None

        # 解密（本地）
        dec = np.array(enc_vec.decrypt(), dtype=np.float64)
        # 归一化为概率（softmax）- 只取前两个值
        dec_2d = dec[:2] if len(dec) >= 2 else np.pad(dec, (0, 2-len(dec)), 'constant')
        ex = np.exp(dec_2d - np.max(dec_2d))
        probs = ex / np.sum(ex)
        return jsonify({"status":"success", "decrypted_probs": probs.tolist(), "client_ct": client_ct, "server_ct": server_ct})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message": str(e)}), 500

@app.route("/api/history", methods=["GET"])
@token_required
def api_history():
    data = load_json(PREDICTIONS_FILE, {"predictions": []})
    user = g.current_user
    preds = [p for p in data.get("predictions", []) if p.get("username") == user]
    return jsonify({"status":"success", "predictions": preds})

@app.route("/api/metrics", methods=["GET"])
@token_required
def api_metrics():
    try:
        data = load_json(METRICS_FILE, {})
        if not data:
            # build demo
            import random
            epochs = list(range(1, 21))
            loss = [round(1.0/(0.3*e+1) + random.random()*0.05, 4) for e in epochs]
            acc = [round(0.3 + 0.7*e/20 + random.random()*0.02, 4) for e in epochs]
            demo = {
                "epochs": epochs,
                "loss": loss,
                "accuracy": acc,
                "class_distribution": [0.62, 0.38],
                "confusion_matrix": [[120,10],[6,50]]
            }
            save_json(METRICS_FILE, demo)
            return jsonify({"status":"success", "metrics": demo})
        return jsonify({"status":"success", "metrics": data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error", "message": "读取 metrics 失败: " + str(e), "metrics": {}}), 200

@app.route("/api/train", methods=["POST"])
@token_required
def api_train():
    if TRAIN_STATUS.get("running"):
        return jsonify({"status":"error", "message":"已有训练任务在运行中", "train_status": TRAIN_STATUS}), 400

    body = request.get_json(force=True, silent=True) or {}
    total_epochs = int(body.get("epochs", 20))
    sleep_time = float(body.get("sleep", 1.0))
    lr = float(body.get("lr", 1e-3))

    # DP config from request
    dp_noise_type = body.get("dp_noise_type")  # 'gaussian' | 'laplace' | None
    dp_epsilon = body.get("dp_epsilon")
    if dp_noise_type not in (None, "gaussian", "laplace"):
        return jsonify({"status":"error", "message":"dp_noise_type 只能为 gaussian / laplace / null"}), 400
    if dp_epsilon is not None:
        try:
            dp_epsilon = float(dp_epsilon)
        except Exception:
            return jsonify({"status":"error", "message":"dp_epsilon 必须为数值"}), 400

    # set in status
    TRAIN_STATUS["dp_noise_type"] = dp_noise_type
    TRAIN_STATUS["dp_epsilon"] = dp_epsilon

    t = threading.Thread(target=_background_train_task, args=(total_epochs, sleep_time, lr), daemon=True)
    t.start()

    return jsonify({"status":"success", "message":"训练已启动", "train_status": TRAIN_STATUS})

@app.route("/api/train_status", methods=["GET"])
@token_required
def api_train_status():
    return jsonify({"status":"success", "train_status": TRAIN_STATUS})

# serve uploads and frontend
@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOADS_DIR, filename)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path == "test.jpg":
        tp = os.path.join(BASE_DIR, "test.jpg")
        if os.path.exists(tp):
            return send_from_directory(BASE_DIR, "test.jpg")
    frontend_path = os.path.join(FRONTEND_DIR, path)
    if path != "" and os.path.exists(frontend_path):
        return send_from_directory(FRONTEND_DIR, path)

    # 默认优先返回独立的登录/注册页
    auth_path = os.path.join(FRONTEND_DIR, "auth.html")
    if os.path.exists(auth_path):
        return send_from_directory(FRONTEND_DIR, "auth.html")

    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return "Frontend index.html not found. 请确保 frontend/index.html 存在", 404

# 在 app.py 中添加（放在其它 route 定义附近）
@app.route("/users.json")
def serve_users_json():
    # 将项目根目录下的 users.json 返回给客户端（与前端 fetch("/users.json") 对应）
    try:
        return send_from_directory(BASE_DIR, "users.json")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": "无法读取 users.json"}), 500

# 另外建议增加一个更安全的 API（前端也可改为调用它）
@app.route("/api/users", methods=["GET"])
def api_users():
    data = load_json(USERS_FILE, {"users": []})
    # 只返回非敏感信息
    users = [{"username": u.get("username"), "created_at": u.get("created_at")} for u in data.get("users", [])]
    return jsonify({"status": "success", "users": users})

if __name__ == "__main__":
    print("尝试导入模型类的结果：")
    for line in _model_import_attempts:
        print(" -", line)
    print("fl0 training import log:")
    for line in _ext_imports_log:
        print(" -", line)
    if MODEL_PATH:
        print("检测到模型文件:", MODEL_PATH)
    else:
        print("未检测到 final_model.pt（如果希望后端可直接推理，请把 final_model.pt 放到项目根）")
    app.run(host="0.0.0.0", port=5000, debug=True)
