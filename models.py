import torch
import torch.nn as nn
import torchvision.models as models
import timm

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        if input_channels == 1:
            # Average pre-trained weights for 1-channel input to preserve features
            old_conv = self.model.conv1
            new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            self.model.conv1 = new_conv
        
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

        if input_channels == 1:
            old_conv = self.model.features[0]
            new_conv = nn.Conv2d(
                1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
            )
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data
            self.model.features[0] = new_conv

        for param in self.model.parameters():
            param.requires_grad = False

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

        if input_channels == 1:
            old_conv = self.model.features[0][0]
            new_conv = nn.Conv2d(
                1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None
            )
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data
            self.model.features[0][0] = new_conv

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        return x.view(x.size(0), -1)
    

class ViTBase(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        # timm is magic. It automatically handles adapting the pre-trained weights 
        # from 3 channels to 1 channel if input_channels=1. No manual math needed!
        # We are using 'vit_tiny_patch16_224' (only ~5 million parameters instead of 86 million)
        self.model = timm.create_model(
            'vit_tiny_patch16_224', 
            pretrained=True, 
            in_chans=input_channels, 
            num_classes=num_classes
        )

        # Freeze everything EXCEPT the final classification head
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        # timm has a built-in method to extract features before the classification head
        features = self.model.forward_features(x)
        
        # forward_features usually returns the sequence of tokens (Batch, Tokens, Features). 
        # We just want the 0th token (the CLS token) for our t-SNE plot.
        if features.dim() == 3:
            return features[:, 0]
        return features


class SimpleMLP(nn.Module):
    def __init__(self, input_size=1024, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        
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
        x = self.flatten(x)
        return self.features(x)