import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # Simple convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: conv (3 -> 16), activation, pooling (224x224 -> 112x112)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: conv (16 -> 32), activation, pooling (112x112 -> 56x56)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: conv (32 -> 64), activation, pooling (56x56 -> 28x28)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5)  # 5 classes: Ragdoll, Singapura, Persian, Sphynx, Pallas
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_baseline_model():
    return BaselineCNN()

if __name__ == "__main__":
    # Check the output shape
    model = get_baseline_model()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Baseline model output shape:", out.shape)  # should be [2, 5]
