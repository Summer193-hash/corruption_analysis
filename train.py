from tqdm import tqdm
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
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training Phase
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        # Wrap train_loader with tqdm for a beautiful progress bar!
        train_loop = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_loop:
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
            
            # Update the progress bar with current loss and accuracy
            current_acc = 100 * correct_train / total_train
            train_loop.set_postfix(loss=loss.item(), acc=f"{current_acc:.2f}%")
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        # Wrap val_loader with tqdm too!
        val_loop = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct_val / total_val
        
        # Print the final epoch summary
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

    # 1. Base settings
    if args.dataset == 'fmnist':
        num_classes, input_channels = 10, 1
    elif args.dataset == 'cifar10':
        num_classes, input_channels = 10, 3
    elif args.dataset == 'imagenet100':
        num_classes, input_channels = 100, 3

    # 2. Dynamic Image Sizing logic
    if args.model == 'vit' or args.dataset == 'imagenet100':
        target_image_size = 224
        optimal_batch_size = 64  # 6GB VRAM safe zone
    else:
        target_image_size = None # Datasets will default to 32x32
        optimal_batch_size = 256 # Speed demon zone

    # Calculate input_size for MLP based on whatever the dataloader will output
    actual_size = 224 if target_image_size == 224 else 32
    input_size = actual_size * actual_size * input_channels

    # -------- CLEAN VALIDATION --------
    print(f"\n--- CLEAN VALIDATION ---")
    train_loader_clean, val_loader_clean, _, _ = get_dataloaders(
        args.dataset, batch_size=optimal_batch_size, corruption_name=None, image_size=target_image_size
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
        model_clean = ViTBase(num_classes=num_classes, input_channels=input_channels)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_clean.parameters(), lr=0.001)

    train_model(
        model_clean, train_loader_clean, val_loader_clean, criterion,
        optimizer, device, epochs=args.epochs, save_dir=save_dir, prefix="clean"
    )

    # -------- CORRUPTED VALIDATION --------
    print(f"\n--- CORRUPTED VALIDATION ---")
    train_loader_corr, val_loader_corr, _, _ = get_dataloaders(
        args.dataset, batch_size=optimal_batch_size, corruption_name='gaussian_noise',
        severity=2, image_size=target_image_size
    )

    if args.model == 'mlp':
        model_corr = SimpleMLP(input_size=input_size, num_classes=num_classes)
    elif args.model == 'resnet':
        model_corr = ResNet18(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'vgg':
        model_corr = VGG11(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'convnext':
        model_corr = ConvNeXtTiny(num_classes=num_classes, input_channels=input_channels)
    elif args.model == 'vit':
        model_corr = ViTBase(num_classes=num_classes, input_channels=input_channels)

    optimizer_corr = optim.Adam(model_corr.parameters(), lr=0.001)

    train_model(
        model_corr, train_loader_corr, val_loader_corr, criterion,
        optimizer_corr, device, epochs=args.epochs, save_dir=save_dir, prefix="corr"
    )