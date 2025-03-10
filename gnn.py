import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define dataset paths
DATASET_PATH = r"E:\TamilBrahmi Dataset\Agilan\processed_images\sorted_images"

# ✅ Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# ✅ Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# ✅ Mapping labels to folder names
label_map = {idx: class_name for class_name, idx in dataset.class_to_idx.items()}

# ✅ Split dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# ✅ Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ✅ Define Graph Neural Network
class GNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# ✅ Initialize Model
num_classes = len(label_map)
model = GNNClassifier(num_classes).to(device)

# ✅ Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# ✅ Training Loop
num_epochs = 2

def train():
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            edge_index = torch.randint(0, images.size(0), (2, images.size(0))).to(device)  # Random graph structure
            batch = torch.arange(images.size(0), device=device)
            
            optimizer.zero_grad()
            outputs = model(images, edge_index, batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)
        
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        
        scheduler.step()

if __name__ == "__main__":
    train()
    # ✅ Save the trained model
    torch.save(model.state_dict(), "gnn_tamilbrahmi.pth")
    print("Training complete. Model saved.")
