import cv2
import torch
import numpy as np
from PIL import Image

# 导入模型和预处理逻辑
from transfer_model import get_transfer_model
from dataset import get_transforms

# 这是我们要识别的 5 种猫
BREEDS = ["Ragdoll", "Singapura", "Persian", "Sphynx", "Pallas"]

def main():
    # 1. 询问并配置视频流 URL
    print("==================================================")
    print("🚀 Great Cat Census - Live Inference Mission Control")
    print("==================================================")
    video_url = input("请输入小车的视频流 URL (如果用电脑摄像头请直接输入 0): ")
    
    if video_url.strip() == "0":
        video_source = 0
    else:
        video_source = video_url.strip()
        
    # 2. 加载 5 分类模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Loading 5-class Transfer Model on {device}...")
    model = get_transfer_model("mobilenet_v2", freeze_backbone=True)
    
    try:
        model.load_state_dict(torch.load("models/best_transfer.pth", map_location=device))
    except FileNotFoundError:
        print("[ERROR] 找不到 models/best_transfer.pth！请先运行 5 分类模型的训练。")
        return
        
    model = model.to(device)
    model.eval()
    
    # 获取验证集的 Transform (含 Resize 和 Normalize)
    transform = get_transforms(is_train=False)
    
    # 3. 连接视频流
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] 无法连接到视频流: {video_source}")
        return
        
    print(f"[INFO] 成功连接视频流！按 'q' 键退出监控。")
    
    # 4. 实时推理循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 无法读取画面，可能网络中断，尝试重连...")
            continue
            
        # 画面从 BGR 转换到 RGB 供 PIL 处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # 预处理
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            # 应用 Softmax 获取置信度概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_breed = BREEDS[predicted_idx.item()]
        conf_percent = confidence.item() * 100
        
        # 5. 在画面上绘制结果
        label_text = f"Breed: {predicted_breed} ({conf_percent:.1f}%)"
        
        # 如果置信度低于 50%，我们可能认为没看清或者不是猫
        color = (0, 255, 0) if conf_percent > 50 else (0, 0, 255)
        
        cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # 显示结果
        cv2.imshow("Mission Control - Great Cat Census", frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
