import torch
import torch.nn as nn
import torchvision.models as models

def get_transfer_model(model_name="mobilenet_v2", freeze_backbone=True):
    """
    加载预训练模型，冻结骨干网络权重，并重写分类头。
    """
    if model_name == "mobilenet_v2":
        # 加载 MobileNetV2
        try:
            # 兼容新版 torchvision
            from torchvision.models import MobileNet_V2_Weights
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except ImportError:
            # 兼容老版 torchvision
            model = models.mobilenet_v2(pretrained=True)
            
        if freeze_backbone:
            # 冻结所有参数权重
            for param in model.parameters():
                param.requires_grad = False
                
        # MobileNetV2 的最后一层结构是 model.classifier，类型为 nn.Sequential
        # 其中包含 nn.Dropout 和 nn.Linear(1280, 1000)
        # 我们用一个新的线性层替换最后一层，使其输出为 2 个类别
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        
    elif model_name == "resnet50":
        # 加载 ResNet50
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except ImportError:
            model = models.resnet50(pretrained=True)
            
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # ResNet50 的最后一层是 fc 层 (nn.Linear(2048, 1000))
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'mobilenet_v2' or 'resnet50'.")
        
    return model

if __name__ == "__main__":
    # 测试 MobileNetV2
    model_mn = get_transfer_model("mobilenet_v2", freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    out = model_mn(x)
    print("MobileNetV2 output shape:", out.shape)
    
    # 检查哪些层有梯度，确认主干网络已被冻结
    trainable_params = [name for name, param in model_mn.named_parameters() if param.requires_grad]
    print("\nTrainable parameters:")
    print(trainable_params)  # 应该只有 classifier.1.weight 和 classifier.1.bias
