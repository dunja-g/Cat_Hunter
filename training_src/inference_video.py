import cv2
import torch
import numpy as np
from PIL import Image

# Import the model and preprocessing
from transfer_model import get_transfer_model
from dataset import get_transforms

# The 5 cat breeds to recognise (same order as preprocess.py)
BREEDS = ["Ragdoll", "Singapura", "Persian", "Sphynx", "Pallas"]

def main():
    # 1. Ask for the video stream URL
    print("==================================================")
    print("🚀 Great Cat Census - Live Inference Mission Control")
    print("==================================================")
    video_url = input("Enter the car's video stream URL (or 0 to use this computer's webcam): ")
    
    if video_url.strip() == "0":
        video_source = 0
    else:
        video_source = video_url.strip()
        
    # 2. Load the 5-class model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Loading 5-class Transfer Model on {device}...")
    model = get_transfer_model("resnet50", freeze_backbone=True)
    
    try:
        model.load_state_dict(torch.load("models/best_transfer.pth", map_location=device))
    except FileNotFoundError:
        print("[ERROR] models/best_transfer.pth not found. Train the 5-class model first.")
        return
        
    model = model.to(device)
    model.eval()
    
    # Use the validation transform (Resize + Normalize)
    transform = get_transforms(is_train=False)
    
    # 3. Connect to the video stream
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot connect to video stream: {video_source}")
        return
        
    print(f"[INFO] Connected to the video stream. Press 'q' to quit.")
    
    # 4. Real-time inference loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Could not read a frame (network interruption?), retrying...")
            continue
            
        # Convert the frame from BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Preprocess
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            # Softmax to get class probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_breed = BREEDS[predicted_idx.item()]
        conf_percent = confidence.item() * 100
        
        # 5. Draw the result on the frame
        label_text = f"Breed: {predicted_breed} ({conf_percent:.1f}%)"
        
        # Below 50% confidence we treat the frame as unclear or not a cat
        color = (0, 255, 0) if conf_percent > 50 else (0, 0, 255)
        
        cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Show the result
        cv2.imshow("Mission Control - Great Cat Census", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
