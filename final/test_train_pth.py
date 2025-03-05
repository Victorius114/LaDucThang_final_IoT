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
dataset = FaceDataset('train_set', transform=transform)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

# Tải mô hình InceptionResnetV1
device = torch.device('cuda')
model = InceptionResnetV1(pretrained='vggface2').to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward & Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tính loss và accuracy
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), 'face_recognition_model.pth')
print("Training finished.")

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

# Vẽ biểu đồ hồi quy tuyến tính trên embeddings
def plot_regression(train_embeddings, train_labels):
    X_train = train_embeddings[:, 0].reshape(-1, 1)  # Lấy một feature từ embeddings
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

# Tạo đối tượng FaceDataset cho tập kiểm tra
test_dataset = FaceDataset('train_set', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Lấy embeddings từ tập huấn luyện và kiểm tra
train_embeddings, train_labels = get_embeddings(model, dataloader, device)
test_embeddings, test_labels = get_embeddings(model, test_dataloader, device)

# So sánh embeddings giữa tập kiểm tra và huấn luyện
distances = pairwise_distances_argmin_min(test_embeddings, train_embeddings)
test_preds = distances[0]

# Tính accuracy trên tập test
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

# Gọi hàm vẽ biểu đồ sau khi lấy embeddings
plot_regression(train_embeddings, train_labels)