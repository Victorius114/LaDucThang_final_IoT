from facenet_pytorch import InceptionResnetV1
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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

# Tạo dataset và dataloader
dataset = FaceDataset(r'dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Tải mô hình InceptionResnetV1
device = torch.device('cuda')
model = InceptionResnetV1(pretrained='vggface2').to(device)

# Lấy embeddings
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

# Dataset kiểm tra
test_dataset = FaceDataset(r'dataset', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Lấy embeddings từ tập huấn luyện và kiểm tra
train_embeddings, train_labels = get_embeddings(model, dataloader, device)
test_embeddings, test_labels = get_embeddings(model, test_dataloader, device)

# CHUẨN HÓA dữ liệu để hồi quy hoặc phân loại chính xác hơn
scaler = StandardScaler()
train_embeddings = scaler.fit_transform(train_embeddings)
test_embeddings = scaler.transform(test_embeddings)

# Phân loại bằng k-NN (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_embeddings, train_labels)
test_preds = knn.predict(test_embeddings)

# Tính độ chính xác
accuracy = accuracy_score(test_labels, test_preds)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# PCA để giảm chiều dữ liệu
pca = PCA(n_components=2)  # Giảm về 2 chiều để trực quan hóa
train_embeddings_2d = pca.fit_transform(train_embeddings)
test_embeddings_2d = pca.transform(test_embeddings)

# Vẽ biểu đồ phân bố embeddings sau khi giảm chiều
plt.figure(figsize=(8, 6))
plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_labels, cmap='jet', alpha=0.6, label="Train Data")
plt.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], c=test_labels, cmap='coolwarm', marker='x', label="Test Data")
plt.colorbar()
plt.title("PCA Visualization of Face Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

import joblib

# Lưu model Facenet (chỉ lưu state_dict để tiết kiệm dung lượng)
torch.save(model.state_dict(), 'facenet_model.pth')

# Lưu k-NN classifier
joblib.dump(knn, 'knn_classifier.pkl')

# Lưu StandardScaler để chuẩn hóa dữ liệu sau này
joblib.dump(scaler, 'scaler.pkl')

# Lưu PCA nếu muốn giảm chiều dữ liệu khi dùng lại
joblib.dump(pca, 'pca.pkl')

print("Đã lưu mô hình và các thành phần liên quan.")

