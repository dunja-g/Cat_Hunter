import torch
import torch.nn as nn
import torchvision.models as models

def get_transfer_model(model_name="mobilenet_v2", freeze_backbone=True):
    """
    Load a pretrained model, optionally freeze the backbone, and replace the classification head.
    """
    if model_name == "mobilenet_v2":
        # Load MobileNetV2
        try:
            # Newer torchvision API
            from torchvision.models import MobileNet_V2_Weights
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except ImportError:
            # Older torchvision API
            model = models.mobilenet_v2(pretrained=True)
            
        if freeze_backbone:
            # Freeze all weights
            for param in model.parameters():
                param.requires_grad = False
                
        # MobileNetV2's last block is model.classifier, an nn.Sequential
        # containing nn.Dropout and nn.Linear(1280, 1000)
        # Replace the last layer with a new linear layer that outputs 5 classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 5)
        
    elif model_name == "convnext_tiny":
        # Load ConvNeXt-Tiny
        try:
            from torchvision.models import ConvNeXt_Tiny_Weights
            model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        except ImportError:
            model = models.convnext_tiny(pretrained=True)
            
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # ConvNeXt's last block is classifier (LayerNorm, Flatten, Linear)
        # Replace the final Linear layer
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 5)
        
    elif model_name == "resnet50":
        # Load ResNet50
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except ImportError:
            model = models.resnet50(pretrained=True)
            
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        # ResNet50's last layer is fc (nn.Linear(2048, 1000))
        # Add Dropout to reduce overfitting
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 5)
        )
        
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'mobilenet_v2', 'resnet50', or 'convnext_tiny'.")
        
    return model

if __name__ == "__main__":
    # Test MobileNetV2
    model_mn = get_transfer_model("mobilenet_v2", freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    out = model_mn(x)
    print("MobileNetV2 output shape:", out.shape)
    
    # Check which layers have gradients, to confirm the backbone is frozen
    trainable_params = [name for name, param in model_mn.named_parameters() if param.requires_grad]
    print("\nTrainable parameters:")
    print(trainable_params)  # should be only classifier.1.weight and classifier.1.bias
