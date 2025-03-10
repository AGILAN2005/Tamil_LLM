# DETR+GNN

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import DetrForObjectDetection, DetrImageProcessor
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define dataset paths
DATASET_PATH = r"E:\TamilBrahmi Dataset\Agilan\processed_images\sorted_images"

# ✅ Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard RGB normalization
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
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# ✅ Load Pretrained DETR Model
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=len(label_map))
model = model.to(device)

# ✅ Define Loss Function & Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# ✅ Define Graph Neural Network (GNN) for refining predictions
class GNNModel(nn.Module):
    def __init__(self, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GCNConv(num_classes, 128)
        self.conv2 = pyg_nn.GCNConv(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

gnn_model = GNNModel(num_classes=len(label_map)).to(device)
gnn_optimizer = optim.AdamW(gnn_model.parameters(), lr=1e-3, weight_decay=1e-4)

# ✅ Training Loop
num_epochs = 10

def train():
    for epoch in range(num_epochs):
        model.train()
        gnn_model.train()
        train_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds, all_probs = [], [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            probs = torch.softmax(outputs.logits, dim=1)
            _, predicted = probs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            
            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)
        
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        map_score = average_precision_score(all_labels, all_probs, average='weighted')
        
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, mAP: {map_score:.4f}")
        
        # ✅ Validation Step
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_labels, all_preds, all_probs = [], [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(outputs.logits, dim=1)
                _, predicted = probs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().detach().numpy())
        
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        map_score = average_precision_score(all_labels, all_probs, average='weighted')
        
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100. * correct / total:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, mAP: {map_score:.4f}")
        scheduler.step()

if __name__ == "__main__":
    train()
    # ✅ Save the trained model
    torch.save(model.state_dict(), "detr_tamilbrahmi.pth")
    torch.save(gnn_model.state_dict(), "gnn_tamilbrahmi.pth")
    print("Training complete. Models saved.")
