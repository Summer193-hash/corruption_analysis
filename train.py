import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import argparse
from data_loader import get_dataloaders
from models import ResNet18, SimpleMLP, VGG11, ConvNeXtTiny, ViTBase
import warnings

warnings.filterwarnings("ignore")


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, save_dir=".", prefix="clean"):
    model.to(device)
    best_val_acc = 0.0
    history = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        history.append([epoch+1, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc])
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"{prefix}.pth"))
            
    csv_path = os.path.join(save_dir, f"training_log_{prefix}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])
        writer.writerows(history)
        
    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%. Saved to {save_dir}\n")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on specified dataset")
    parser.add_argument('--dataset', type=str, required=True, choices=['fmnist', 'cifar10', 'imagenet100'])
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'resnet', 'vgg', 'convnext', 'vit'])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = os.path.join("results", args.dataset, args.model)

    # IMPORTANT: update input size for resized images
    if args.dataset == 'fmnist':
        input_size, num_classes, input_channels, image_size = 32 * 32 * 1, 10, 1, 32
    elif args.dataset == 'cifar10':
        input_size, num_classes, input_channels, image_size = 128 * 128 * 3, 10, 3, 128
    elif args.dataset == 'imagenet100':
        input_size, num_classes, input_channels, image_size = 224 * 224 * 3, 100, 3, 224

    # -------- CLEAN VALIDATION --------
    print(f"\n--- CLEAN VALIDATION ---")
    train_loader_clean, val_loader_clean, _, _ = get_dataloaders(
        args.dataset, batch_size=16, corruption_name=None
    )

    if args.model == 'mlp':
        model_clean = SimpleMLP(input_size=input_size, num_classes=num_classes)
    elif args.model == 'resnet':
        model_clean = ResNet18(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'vgg':
        model_clean = VGG11(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'convnext':
        model_clean = ConvNeXtTiny(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'vit':
        model_clean = ViTBase(num_classes=num_classes, image_size=image_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_clean.parameters(), lr=0.001)

    train_model(
        model_clean,
        train_loader_clean,
        val_loader_clean,
        criterion,
        optimizer,
        device,
        epochs=args.epochs,
        save_dir=save_dir,
        prefix="clean"
    )

    # -------- CORRUPTED VALIDATION --------
    print(f"\n--- CORRUPTED VALIDATION ---")
    train_loader_corr, val_loader_corr, _, _ = get_dataloaders(
        args.dataset,
        batch_size=64,
        corruption_name='gaussian_noise',
        severity=2
    )

    # FIXED: this was wrong in your code
    if args.model == 'mlp':
        model_corr = SimpleMLP(input_size=input_size, num_classes=num_classes)
    elif args.model == 'resnet':
        model_corr = ResNet18(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'vgg':
        model_corr = VGG11(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'convnext':
        model_corr = ConvNeXtTiny(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'vit':
        model_corr = ViTBase(num_classes=num_classes, image_size=image_size)

    optimizer_corr = optim.Adam(model_corr.parameters(), lr=0.001)

    train_model(
        model_corr,
        train_loader_corr,
        val_loader_corr,
        criterion,
        optimizer_corr,
        device,
        epochs=args.epochs,
        save_dir=save_dir,
        prefix="corr"
    )