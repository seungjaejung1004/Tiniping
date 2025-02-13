
import os
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import tarfile
import math
from collections import Counter

# 모델 파일과 데이터셋 경로 설정
model_path = "ping01.pth"
dataset_path = "ifw-funneled.tgz"
extracted_dir = "lfw_images"

# 1. 데이터셋 압축 해제
if not os.path.exists(extracted_dir):
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall(extracted_dir)

# 2. 이미지 파일 경로 로드
all_images = []
for root, _, files in os.walk(extracted_dir):
    for file in files:
        if file.endswith(".jpg"):
            all_images.append(os.path.join(root, file))

# 3. 이미지 전처리 및 샘플링
sampled_images = random.sample(all_images, 500)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 정규화 해제용 변환
unnormalize = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
])

# 4. MPS 디바이스 설정
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 5. 모델 로드 및 디바이스로 이동
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()


# 6. 예측 및 시각화
def show_predictions(images, predictions, save_path="predictions_output.png"):
    num_images = len(images)
    cols = 10
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2.5))
    axes = axes.flatten()

    for img_tensor, pred, ax in zip(images, predictions, axes):
        img = unnormalize(img_tensor)
        img = transforms.ToPILImage()(img)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{pred}", fontsize=8)

    for ax in axes[len(images):]:
        ax.remove()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# 결과 저장 변수
images, predicted_classes = [], []

with torch.no_grad():
    for img_path in sampled_images:
        image = Image.open(img_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)
        output = model(input_image)

        pred_prob = torch.nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(pred_prob, dim=1).item()
        confidence = pred_prob[0][pred_class].item()

        images.append(transform(image))
        predicted_classes.append(f"Class {pred_class}")  # 확률 정보 제외하고 클래스 번호만 저장

# 각 클래스별 카운트
class_counts = Counter(predicted_classes)

# 클래스별 카운트를 원하는 형식으로 출력
print("Class Counts:")
for class_name, count in sorted(class_counts.items()):
    # 확률을 제외한 클래스 번호만 출력
    class_num = class_name.split()[1]  # "Class X"에서 X만 추출
    print(f"class{class_num}: {count}개")

# 결과 시각화 및 저장
show_predictions(images, predicted_classes, save_path="predictions_output.png")