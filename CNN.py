import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from torch.utils.data import DataLoader, Subset

# 固定模型初始化等随机过程；数据划分另用独立 generator。
torch.manual_seed(42)

train_transform = transforms.Compose([
transforms.RandomAffine(
        degrees=15,              # 旋转：-15°～15°
        translate=(0.15, 0.15),   # 横纵平移：最多约 4 个像素
        scale=(0.9, 1.1),         # 缩放：原尺寸的 90%～110%
        shear=5,                 # 剪切：模拟字体倾斜
        interpolation=InterpolationMode.BILINEAR,
        fill=0                   # 空白区域填黑色
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 1. 相同的原始数据，分别配置不同变换
train_source = datasets.MNIST(
    './dataset',
    train=True,
    download=True,
    transform=train_transform
)

val_source = datasets.MNIST(
    './dataset',
    train=True,
    download=True,
    transform=eval_transform
)

# 2. 使用同一份随机索引划分，确保两个集合没有重叠
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(train_source), generator=generator).tolist()

train_dataset = Subset(train_source, indices[:55000])
val_dataset = Subset(val_source, indices[55000:])

test_loader = DataLoader(
datasets.MNIST('./dataset', train=False, download=True, transform=eval_transform),
batch_size=120, shuffle=False
)
train_loader = DataLoader(
    train_dataset, batch_size=120, shuffle=True
)

val_loader = DataLoader(
    val_dataset, batch_size=120, shuffle=False
)
images, labels = next(iter(train_loader))

##print("标签形状：", labels.shape)  # [64]
##print("前十个标签：", labels[:10])
#print("图片类型：", images.dtype)  # torch.float32
#print("标签类型：", labels.dtype)  # torch.int64

#print("训练集大小：", len(train_loader.dataset))  # 55000
#print("验证集大小：", len(val_loader.dataset))    # 5000
#print("测试集大小：", len(test_loader.dataset))   # 10000


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 无池化：宽高依次为 28 → 24 → 20 → 16 → 12 → 8
            nn.Conv2d(1, 32, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),

            nn.Conv2d(32, 64, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),

            nn.Conv2d(64, 96, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm2d(96),
            nn.Sigmoid(),

            nn.Conv2d(96, 128, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),

            nn.Conv2d(128, 160, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm2d(160),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160 * 8 * 8, 128),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MNISTCNN().to(device)

images, labels = next(iter(train_loader))
images = images.to(device)

model.eval()
with torch.no_grad():
    outputs = model(images)

print("运行设备：", device)
print("输入形状：", images.shape)    # [120, 1, 28, 28]
print("输出形状：", outputs.shape)   # [120, 10]
print("参数数量：", sum(p.numel() for p in model.parameters()))

# 正式训练前切回训练模式
model.train()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

        if batch_index % 100 == 0:
            print(f"  批次 {batch_index}/{len(loader)}，loss={loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total, total - correct

def main():
    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _ = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"第 {epoch}/{epochs} 轮 | "
            f"训练损失：{train_loss:.4f}，准确率：{train_acc:.2%} | "
            f"验证损失：{val_loss:.4f}，准确率：{val_acc:.2%}"
        )

        # 保存验证集上表现最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_cnn.pth")

    # 加载最佳模型，在测试集上评估
    model.load_state_dict(
        torch.load("best_cnn.pth", map_location=device, weights_only=True)
    )
    test_loss, test_acc, errors = evaluate(
        model, test_loader, criterion, device
    )
    print(
        f"测试损失：{test_loss:.4f}，"
        f"准确率：{test_acc:.2%}，错误数量：{errors}"
    )


if __name__ == "__main__":
    main()