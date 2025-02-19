from facenet_pytorch import InceptionResnetV1
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from sklearn.metrics import accuracy_score, pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Tiền xử lý ảnh
def preprocess_image(image_path, required_size=(160, 160)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Không thể đọc ảnh từ {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, required_size)
    return image_resized

# Lớp Dataset cho dữ liệu khuôn mặt
class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for idx, person in enumerate(os.listdir(data_dir)):
            person_folder = os.path.join(data_dir, person)
            if os.path.isdir(person_folder):
                for image_name in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, image_name)
                    image = preprocess_image(image_path)
                    if image is not None:
                        self.images.append(image_path)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = preprocess_image(image_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

# Các chuyển đổi để chuẩn bị dữ liệu
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Tạo đối tượng FaceDataset
dataset = FaceDataset('E:/Ky2_2024_2025/BT_IoT_CV/dataset2', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Tải mô hình InceptionResnetV1
device = torch.device('cuda')
model = InceptionResnetV1(pretrained='vggface2').to(device)

# Huấn luyện mô hình
model.train()
for epoch in range(30):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Thực hiện huấn luyện ở đây (forward, backward, optimize)
        pass

# Lưu mô hình dưới dạng ONNX
dummy_input = torch.randn(1, 3, 160, 160, device=device)  # Định dạng đầu vào của mô hình
onnx_path = "final/face_recognition_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11, input_names=['input'], output_names=['output'])
print(f"Model saved as {onnx_path}")

# Đánh giá mô hình
def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, target_labels in dataloader:
            inputs = inputs.to(device)
            output = model(inputs)
            embeddings.append(output.cpu().numpy())
            labels.append(target_labels.numpy())
    return np.concatenate(embeddings), np.concatenate(labels)

# Tạo đối tượng FaceDataset cho tập kiểm tra
test_dataset = FaceDataset('dataset', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Lấy embeddings từ tập huấn luyện và kiểm tra
train_embeddings, train_labels = get_embeddings(model, dataloader, device)
test_embeddings, test_labels = get_embeddings(model, test_dataloader, device)

# So sánh embeddings giữa tập kiểm tra và huấn luyện
distances = pairwise_distances_argmin_min(test_embeddings, train_embeddings)
test_preds = distances[0]

# Tính accuracy
accuracy = accuracy_score(test_labels, test_preds)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Vẽ biểu đồ hồi quy tuyến tính
X_train = train_embeddings[:, 0].reshape(-1, 1)  # Chọn một chiều từ embeddings làm feature
y_train = train_labels

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_train)

# Vẽ biểu đồ
plt.scatter(X_train, y_train, color='blue', label='Actual labels')
plt.plot(X_train, y_pred, color='red', label='Linear regression')
plt.title('Linear Regression on Embeddings')
plt.xlabel('Embedding Feature')
plt.ylabel('Labels')
plt.legend()
plt.show()
