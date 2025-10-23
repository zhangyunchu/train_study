# model.py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST

# ===============================
# 一、数据加载（EMNIST Balanced）
# ===============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)
test_data = EMNIST(root="./data", split="balanced", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

classes = list(train_data.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 二、模型定义
# ===============================
class EMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 47)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===============================
# 三、工具函数
# ===============================
def load_model(model_path="./model/emnist_balanced_model.pkl"):
    model = EMNISTModel().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("已加载 EMNIST Balanced 模型。")
    else:
        print("未检测到已训练模型，将使用随机初始化。")
    return model


def train_model(model, train_loader, criterion, optimizer, model_path, progress_callback=None, epoch_callback=None):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    total_epochs = 2
    total_batches = len(train_loader) * total_epochs
    batch_count = 0

    for epoch in range(total_epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            batch_count += 1
            if progress_callback:
                progress_callback(batch_count / total_batches * 100)

        torch.save(model.state_dict(), model_path)
        if epoch_callback:
            epoch_callback(epoch + 1, total_loss / len(train_loader))


def test_model(model, test_loader, progress_callback=None):
    correct, total = 0, 0
    total_batches = len(test_loader)
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            if progress_callback:
                progress_callback((i + 1) / total_batches * 100)

    return correct / total
