from facenet_pytorch import InceptionResnetV1
import torch
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from sklearn.metrics import accuracy_score, pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge


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
dataset = FaceDataset(r'dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Tải mô hình InceptionResnetV1
device = torch.device('cuda')
model = InceptionResnetV1(pretrained='vggface2').to(device)

# Huấn luyện mô hình
model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        pass

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Tính toán loss và accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{30}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# Lưu mô hình dưới dạng ONNX
dummy_input = torch.randn(1, 3, 160, 160, device=device)  # Định dạng đầu vào của mô hình
onnx_path = "face_recognition_model2.onnx"
torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11, input_names=['input'],
                  output_names=['output'])
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
test_dataset = FaceDataset(r'dataset', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Lấy embeddings từ tập huấn luyện và kiểm tra
train_embeddings, train_labels = get_embeddings(model, dataloader, device)
test_embeddings, test_labels = get_embeddings(model, test_dataloader, device)

# So sánh embeddings giữa tập kiểm tra và huấn luyện
distances = pairwise_distances_argmin_min(test_embeddings, train_embeddings)
test_preds = distances[0]

# Chọn 1 chiều từ embeddings làm feature
X_train = train_embeddings[:, 0].reshape(-1, 1)
y_train = train_labels

# Thay vì hồi quy đa thức tuyến tính, dùng Ridge Regression để regularization
degree = 2  # Giảm bậc đa thức để tránh overfitting
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X_train)

model_poly = Ridge(alpha=0.1)  # Ridge Regression với alpha=0.1 để giảm overfitting
model_poly.fit(X_poly, y_train)

# Dự đoán nhãn
y_pred_poly = model_poly.predict(X_poly)

# Sắp xếp dữ liệu theo X_train để vẽ đồ thị mượt mà
sorted_indices = np.argsort(X_train[:, 0])
X_sorted = X_train[sorted_indices]
y_pred_poly_sorted = y_pred_poly[sorted_indices]

# Vẽ biểu đồ
plt.scatter(X_train, y_train, color='blue', label='Actual labels')
plt.plot(X_sorted, y_pred_poly_sorted, color='green', label=f'Polynomial regression (degree={degree})')
plt.title('Polynomial Regression on Embeddings')
plt.xlabel('Embedding Feature')
plt.ylabel('Labels')
plt.legend()
plt.show()