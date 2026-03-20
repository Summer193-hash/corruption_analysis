import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from models import SimpleMLP, ResNet18, VGG11, ConvNeXtTiny, ViTBase
from data_loader import get_dataloaders
import warnings

warnings.filterwarnings("ignore")

def test_accuracy(model, dataloader, device):
    """Calculates actual classification accuracy on a test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def extract_features(model, dataloader, device, num_samples=1000):
    model.eval()
    features_list, labels_list = [], []
    samples_collected = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model.get_features(inputs)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            samples_collected += inputs.size(0)
            if samples_collected >= num_samples: break
            
    return np.concatenate(features_list, axis=0)[:num_samples], np.concatenate(labels_list, axis=0)[:num_samples]

def plot_tsne(features, labels, title, save_path):
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=labels, palette=sns.color_palette("tab10", len(np.unique(labels))), legend=False, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_model(model_type, input_size, num_classes, input_channels, device):
    """Create appropriate model based on model_type."""
    if model_type == 'mlp':
        model = SimpleMLP(input_size=input_size, num_classes=num_classes)
    elif model_type == 'resnet':
        model = ResNet18(num_classes=num_classes, input_channels=input_channels)
    elif model_type == 'vgg':
        model = VGG11(num_classes=num_classes, input_channels=input_channels)
    elif model_type == 'convnext':
        model = ConvNeXtTiny(num_classes=num_classes, input_channels=input_channels)
    elif model_type == 'vit':
        model = ViTBase(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['fmnist', 'cifar10', 'imagenet100'])
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'resnet', 'vgg', 'convnext', 'vit'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.dataset == 'fmnist':
        input_size, num_classes, input_channels = 64 * 64 * 1, 10, 1
    elif args.dataset == 'cifar10':
        input_size, num_classes, input_channels = 64 * 64 * 3, 10, 3
    elif args.dataset == 'imagenet100':
        input_size, num_classes, input_channels = 64 * 64 * 3, 100, 3

    save_dir = os.path.join("results", args.dataset, args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    summary_file_path = os.path.join(save_dir, "evaluation_summary.txt")
    
    # Open summary file to write metrics
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"=== Evaluation Summary for {args.dataset.upper()} ===\n\n")

        # Load testing dataloaders
        _, _, test_loader_clean, test_loader_corr = get_dataloaders(args.dataset, batch_size=64, corruption_name='gaussian_noise', severity=2)

        # 1. Evaluate Clean-Trained Model
        clean_path = os.path.join(save_dir, f'clean.pth')
        if os.path.exists(clean_path):
            model_clean = create_model(args.model, input_size, num_classes, input_channels, device)
            model_clean.load_state_dict(torch.load(clean_path))
            
            print(f"\n--- Model Trained on CLEAN Validation ({args.dataset}) ---")
            clean_acc = test_accuracy(model_clean, test_loader_clean, device)
            robust_acc = test_accuracy(model_clean, test_loader_corr, device)
            
            print(f"Accuracy on Clean Test Set: {clean_acc:.2f}%")
            print(f"Accuracy on Corrupted Test Set (Robustness): {robust_acc:.2f}%")
            
            summary_file.write(f"--- Clean-Trained Model ---\n")
            summary_file.write(f"Clean Test Acc: {clean_acc:.2f}%\n")
            summary_file.write(f"Corrupted Test Acc: {robust_acc:.2f}%\n\n")
            
            f_clean, l_clean = extract_features(model_clean, test_loader_clean, device)
            plot_tsne(f_clean, l_clean, f"Clean Model ({args.dataset})", os.path.join(save_dir, f"tsne_clean.png"))
        else:
            print(f"\n[ERROR] Model checkpoint not found at: {clean_path}")
            print(f"Make sure you run: python train.py --dataset {args.dataset}")

        # 2. Evaluate Corrupted-Trained Model
        corr_path = os.path.join(save_dir, f'corr.pth')
        if os.path.exists(corr_path):
            model_corr = create_model(args.model, input_size, num_classes, input_channels, device)
            model_corr.load_state_dict(torch.load(corr_path))
            
            print(f"\n--- Model Trained on CORRUPTED Validation ({args.dataset}) ---")
            clean_acc_corr = test_accuracy(model_corr, test_loader_clean, device)
            robust_acc_corr = test_accuracy(model_corr, test_loader_corr, device)
            
            print(f"Accuracy on Clean Test Set: {clean_acc_corr:.2f}%")
            print(f"Accuracy on Corrupted Test Set (Robustness): {robust_acc_corr:.2f}%")
            
            summary_file.write(f"--- Corrupted-Trained Model ---\n")
            summary_file.write(f"Clean Test Acc: {clean_acc_corr:.2f}%\n")
            summary_file.write(f"Corrupted Test Acc: {robust_acc_corr:.2f}%\n\n")
            
            f_corr, l_corr = extract_features(model_corr, test_loader_clean, device)
            plot_tsne(f_corr, l_corr, f"Corrupted Model ({args.dataset})", os.path.join(save_dir, f"tsne_corr.png"))
        else:
            print(f"\n[ERROR] Model checkpoint not found at: {corr_path}")