import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
from torchvision.models import resnet18, ResNet18_Weights

# 경로 설정
data_dir = './people'
batch_size = 32
num_classes = 25
num_epochs = 25  # 에포크 수
patience = 3  # 조기 종료를 위한 patience 설정

# MPS 장치 설정
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 데이터 전처리: 데이터 증강 및 정규화
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),  # 다양한 크기로 랜덤 크롭
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),  # 수직 뒤집기 (조금만 적용)
    transforms.RandomRotation(20),  # 회전 각도 증가
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 밝기, 대비, 채도, 색상 변화 추가
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 이미지 이동 추가
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 로드
image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# 훈련 및 검증 데이터셋 분리
train_size = int(0.8 * len(image_dataset))
val_size = len(image_dataset) - train_size
train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 사전 학습된 ResNet 모델 불러오기 (weights 인자 사용)
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# ResNet의 마지막 출력층을 18개의 클래스로 바꾸기 및 드롭아웃 추가
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # 50% 드롭아웃 추가
    nn.Linear(num_ftrs, num_classes)
)
model = model.to(device)

# 손실 함수와 최적화 및 학습률 스케줄러 설정 (L2 정규화 포함)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 스케줄 조정

# 검증 함수
def validate_model(model, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = val_loss / len(val_dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# 모델 훈련 함수 (조기 종료 포함)
train_accuracies = []
val_accuracies = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=15, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = validate_model(model, criterion)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # 조기 종료 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break

# 모델 훈련 시작
train_model(model, criterion, optimizer, scheduler, num_epochs, patience)
# 모델 저장
torch.save(model, 'ping01.pth')