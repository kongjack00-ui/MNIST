from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),           # 将 28×28 图片展开成 784 个数
            nn.Linear(784, 256),     # 输入层 → 隐藏层
            nn.ReLU(),
            nn.Linear(256, 10),      # 隐藏层 → 输出层
            #nn.Sigmoid()            # 每个输出在 0～1 之间
        )

    def forward(self, x):
        return self.network(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # 例如标签 2 转成 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        targets = nn.functional.one_hot(labels, num_classes=10).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()             # 反向传播，计算梯度
        optimizer.step()            # 根据梯度更新参数

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def test(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        targets = nn.functional.one_hot(labels, num_classes=10).float()
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    torch.manual_seed(42)
    epochs = 30
    batch_size = 120
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(__file__).resolve().parent / "dataset"
    transform = transforms.ToTensor()  # 像素值从 0～255 转成 0～1
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BPNet().to(device)
    criterion = nn.CrossEntropyLoss()  # 改成交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"运行设备：{device}，每轮训练 {len(train_loader)} 个 batch")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(
            f"第 {epoch}/{epochs} 轮 | "
            f"训练损失：{train_loss:.4f}，训练准确率：{train_acc:.2%}"
        )

    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f"测试损失：{test_loss:.4f}，测试准确率：{test_acc:.2%}")


if __name__ == "__main__":
    main()
