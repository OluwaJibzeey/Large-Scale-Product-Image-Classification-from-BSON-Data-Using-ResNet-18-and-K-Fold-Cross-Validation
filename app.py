import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, io
from torch.utils.data import DataLoader, Dataset
import bson
import struct
from PIL import Image
import io as pyio
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

# Read category IDs
category_df = pd.read_csv("./data/category_names.csv")
category_ids = category_df["category_id"].unique()
num_classes = len(category_ids)
category_id_to_idx = {cid: idx for idx, cid in enumerate(category_ids)}

# Define transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# BSON reader function with offsets
def read_bson_offsets(file_path):
    offsets = []
    with open(file_path, "rb") as f:
        current_offset = 0
        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break
            length = struct.unpack("<i", length_bytes)[0]
            offsets.append((current_offset, current_offset + length))
            current_offset += length
            f.seek(current_offset)
    return offsets


# Training dataset
class TrainDataset(Dataset):
    def __init__(self, transform, file_path):
        self.transform = transform
        self.file_path = file_path
        self.offsets = read_bson_offsets(file_path)
        self.f = open(file_path, "rb")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        start, end = self.offsets[idx]
        self.f.seek(start)
        length_bytes = self.f.read(4)
        length = struct.unpack("<i", length_bytes)[0]
        doc_bytes = length_bytes + self.f.read(length - 4)
        doc = bson.decode(doc_bytes)
        category_id = doc["category_id"]
        imgs = doc["imgs"]
        if not imgs:
            return self.__getitem__((idx + 1) % len(self))  # Skip invalid
        image_data = imgs[0]["picture"]
        image = Image.open(pyio.BytesIO(image_data)).convert("RGB")
        image = self.transform(image)
        return image, category_id_to_idx[category_id]

    def __del__(self):
        self.f.close()


# Test dataset
class TestDataset(Dataset):
    def __init__(self, transform, file_path):
        self.transform = transform
        self.file_path = file_path
        self.offsets = read_bson_offsets(file_path)
        self.f = open(file_path, "rb")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        start, end = self.offsets[idx]
        self.f.seek(start)
        length_bytes = self.f.read(4)
        length = struct.unpack("<i", length_bytes)[0]
        doc_bytes = length_bytes + self.f.read(length - 4)
        doc = bson.decode(doc_bytes)
        product_id = doc["_id"]
        imgs = doc["imgs"]
        if not imgs:
            return self.__getitem__((idx + 1) % len(self))  # Skip invalid
        image_data = imgs[0]["picture"]
        image = Image.open(pyio.BytesIO(image_data)).convert("RGB")
        image = self.transform(image)
        return image, product_id

    def __del__(self):
        self.f.close()


# Model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_dataset = TrainDataset(transform, "./data/train.bson")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
best_accuracy = 0.0
for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    train_loader_fold = DataLoader(
        train_dataset, batch_size=32, sampler=train_subsampler
    )
    val_loader_fold = DataLoader(train_dataset, batch_size=32, sampler=val_subsampler)

    model.train()
    for epoch in range(5):
        for images, labels in train_loader_fold:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader_fold:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Fold {fold+1}, Accuracy: {accuracy:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")

# Load best model
model.load_state_dict(torch.load("best_model.pth"))

# Make predictions on test set
test_dataset = TestDataset(transform, "./data/test.bson")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
predictions = []
product_ids = []
with torch.no_grad():
    for images, ids in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        product_ids.extend(ids)

# Map predictions back to category_ids
predicted_category_ids = [category_ids[pred] for pred in predictions]

# Create submission file
submission_df = pd.DataFrame(
    {"_id": product_ids, "category_id": predicted_category_ids}
)
submission_df.to_csv("submission.csv", index=False)

print(f"Best 5-fold cross-validation accuracy: {best_accuracy:.4f}")
print("Submission file generated.")
