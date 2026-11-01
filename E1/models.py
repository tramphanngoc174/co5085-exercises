"""E1: ba bộ phân loại. Đầu ra là LOGIT; softmax nằm trong CrossEntropyLoss khi huấn luyện
(muốn xác suất thì gọi torch.softmax(logits, dim=1))."""
import torch.nn as nn


class SoftmaxClassifier(nn.Module):
    """Flatten ảnh -> ánh xạ tuyến tính -> (softmax trong loss). Không có tầng ẩn."""
    def __init__(self, in_channels=1, img_size=28, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_channels * img_size * img_size, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))


class MLP(nn.Module):
    """Đầu vào vẫn là vector flatten; nhiều tầng fully connected + ReLU."""
    def __init__(self, in_channels=1, img_size=28, num_classes=10, hidden=(512, 256)):
        super().__init__()
        layers, d = [nn.Flatten()], in_channels * img_size * img_size
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """2 khối [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2] + head phân loại."""
    def __init__(self, in_channels=1, img_size=28, num_classes=10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(in_channels, 32), block(32, 64))
        feat = 64 * (img_size // 4) ** 2     # 28 -> 14 -> 7  => 64*7*7
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


MODELS = {"softmax": SoftmaxClassifier, "mlp": MLP, "cnn": SimpleCNN}
