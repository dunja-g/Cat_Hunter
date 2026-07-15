import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # 定义简单的卷积特征提取层
        self.features = nn.Sequential(
            # 第一层：卷积 (3 -> 16), 激活, 池化 (224x224 -> 112x112)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层：卷积 (16 -> 32), 激活, 池化 (112x112 -> 56x56)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层：卷积 (32 -> 64), 激活, 池化 (56x56 -> 28x28)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 定义全连接分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5)  # 五分类：Ragdoll, Singapura, Persian, Sphynx, Pallas
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_baseline_model():
    return BaselineCNN()

if __name__ == "__main__":
    # 测试模型维度
    model = get_baseline_model()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Baseline model output shape:", out.shape)  # 应该是 [2, 2]
