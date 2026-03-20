import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        # Adjust first convolutional layer for grayscale or RGB input
        if input_channels == 1:
            # Modify the first conv layer to accept 1 channel input
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        return x.view(x.size(0), -1)
    

class VGG11(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.model = models.vgg11(pretrained=True)

        # Adjust first conv layer for grayscale input
        if input_channels == 1:
            old_conv = self.model.features[0]
            new_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data
            self.model.features[0] = new_conv

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classifier
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.model = models.convnext_tiny(pretrained=True)

        # Adjust first conv layer for grayscale input
        if input_channels == 1:
            old_conv = self.model.features[0][0]
            new_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data
            self.model.features[0][0] = new_conv

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classifier
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        return x.view(x.size(0), -1)
    

class ViTBase(nn.Module):
    def __init__(self, num_classes=10, image_size=224):
        super().__init__()
        image_size = image_size  # ViTBase is typically trained on 224x224 
        self.model = models.vit_b_16(pretrained=True, image_size=224)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classifier head
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        # Extract features before classification head
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # CLS token
        return x[:, 0]


class SimpleMLP(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10):
        # F-MNIST images are 28x28
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        # Standard architecture with 2 hidden layers
        self.features = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        out = self.classifier(x)
        return out

    def get_features(self, x):
        # We need this method later to extract feature maps for the t-SNE plots
        x = self.flatten(x)
        return self.features(x)
    
