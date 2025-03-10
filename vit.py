import os
import torch
import timm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from timm.optim import create_optimizer_v2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# ✅ Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define dataset paths
# DATASET_PATH = r"E:\TamilBrahmi Dataset\Agilan\processed_images\sorted_images"
DATASET_PATH = r"E:\TamilBrahmi Dataset\Agilan\processed_images\sorted_images\U111017_U11038"
# ✅ Image transformations (ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

def load_data():
    """Loads dataset, splits it, and creates data loaders."""
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, num_classes

def create_model(num_classes):
    """Loads a pretrained ViT model and modifies it for classification."""
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    return model.to(device)

def train_model(model, train_loader, val_loader, num_epochs=2):
    """Trains and validates the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer_v2(model, "adamw", lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item(), acc=100. * float(correct) / float(total))

        # ✅ Training Metrics
        train_acc = 100. * float(correct) / float(total)
        train_precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        train_recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
        train_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

        print(f"\n🔹 Epoch {epoch+1} Training Metrics:")
        print(f"   Loss: {train_loss / len(train_loader):.4f}")
        print(f"   Accuracy: {train_acc:.2f}%")
        print(f"   Precision: {train_precision:.4f}")
        print(f"   Recall: {train_recall:.4f}")
        print(f"   F1-Score: {train_f1:.4f}")

        # ✅ Validation Step
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # ✅ Validation Metrics
        val_acc = 100. * float(correct) / float(total)
        val_precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        val_recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
        val_f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

        print(f"\n✅ Validation Results (Epoch {epoch+1}):")
        print(f"   Loss: {val_loss / len(val_loader):.4f}")
        print(f"   Accuracy: {val_acc:.2f}%")
        print(f"   Precision: {val_precision:.4f}")
        print(f"   Recall: {val_recall:.4f}")
        print(f"   F1-Score: {val_f1:.4f}")

        scheduler.step()

    # ✅ Save model
    torch.save(model.state_dict(), "vit_tamilbrahmi.pth")
    print("\n✅ Training complete. Model saved successfully!")

# ✅ Main function (Avoids multiprocessing issues)
if __name__ == "__main__":
    print(f"Using device: {device}")  # ✅ Only prints once!
    
    train_loader, val_loader, num_classes = load_data()
    model = create_model(num_classes)
    train_model(model, train_loader, val_loader, num_epochs=2)
