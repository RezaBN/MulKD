# models.py
"""
This file contains all model architecture definitions, including ResNets for CIFAR-10
and lightweight models like MobileNetV2 and ShuffleNetV2. It also includes helper
functions to extract outputs and feature dimensions from these models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# --- ResNet Architectures (for CIFAR) ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # This implementation is specific to CIFAR-style ResNets with 3 stages
        if len(num_blocks) == 3:
            self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
            self.penultimate_dim = 64 * block.expansion
        else:
             raise ValueError("num_blocks for CIFAR ResNet should be a list of 3 integers")

        self.linear = nn.Linear(self.penultimate_dim, num_classes)
        # Add a flag to indicate native support for returning features
        self.supports_return_features = True

    def _make_layer(self, block, planes, num_blocks_in_stage, stride):
        strides = [stride] + [1] * (num_blocks_in_stage - 1)
        layers = []
        for s_val in strides:
            layers.append(block(self.in_planes, planes, s_val))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        pool_size = out.shape[-1] # Adaptive pooling based on feature map size
        features_for_crd = F.avg_pool2d(out, kernel_size=pool_size)
        features_for_crd = features_for_crd.view(features_for_crd.size(0), -1)

        logits = self.linear(features_for_crd)

        return (logits, features_for_crd) if return_features else logits

# --- ResNet Factory Functions ---
def ResNet8_cifar(num_classes=10): return ResNet(BasicBlock, [1, 1, 1], num_classes)
def ResNet20_cifar(num_classes=10): return ResNet(BasicBlock, [3, 3, 3], num_classes)
def ResNet32_cifar(num_classes=10): return ResNet(BasicBlock, [5, 5, 5], num_classes)
def ResNet38_cifar(num_classes=10): return ResNet(BasicBlock, [6, 6, 6], num_classes)
def ResNet44_cifar(num_classes=10): return ResNet(BasicBlock, [7, 7, 7], num_classes)
def ResNet50_cifar_approx(num_classes=10): return ResNet(BasicBlock, [8, 8, 8], num_classes)
def ResNet56_cifar(num_classes=10): return ResNet(BasicBlock, [9, 9, 9], num_classes)
def ResNet110_cifar(num_classes=10): return ResNet(BasicBlock, [18, 18, 18], num_classes)


# --- Lightweight Models ---
def MobileNetV2_paper(num_classes=10, pretrained=False):
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.supports_return_features = False # Handled by get_model_output
    model.penultimate_dim = model.last_channel
    return model

def ShuffleNetV2_paper(num_classes=10, pretrained=False):
    model = torchvision.models.shufflenet_v2_x1_0(weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None)
    original_fc_in_features = model.fc.in_features
    model.fc = nn.Linear(original_fc_in_features, num_classes)
    model.penultimate_dim = original_fc_in_features
    model.supports_return_features = False # Handled by get_model_output
    return model


# --- Model Output Helpers ---
def get_model_output(model, inputs, for_crd=False):
    """
    A unified function to get logits and feature representations from any model.
    This handles differences between custom ResNets and torchvision models.
    """
    logits, features = None, None

    if hasattr(model, 'supports_return_features') and model.supports_return_features:
        # This branch is for our custom ResNet models
        logits, features = model(inputs, return_features=True)

    elif isinstance(model, torchvision.models.MobileNetV2):
        # Specific handling for torchvision MobileNetV2
        extracted_feats_module = model.features(inputs)
        # Global average pooling before the classifier
        features = F.adaptive_avg_pool2d(extracted_feats_module, (1, 1)).reshape(extracted_feats_module.shape[0], -1)
        logits = model.classifier(features)

    elif isinstance(model, torchvision.models.ShuffleNetV2):
        # Specific handling for torchvision ShuffleNetV2
        x = model.conv1(inputs); x = model.maxpool(x); x = model.stage2(x)
        x = model.stage3(x); x = model.stage4(x)
        extracted_feats_module = model.conv5(x)
        features = extracted_feats_module.mean([2, 3]) # Global average pooling
        logits = model.fc(features)

    else:
        # Fallback for other models, may not provide features for CRD
        logits = model(inputs)
        if for_crd:
            print(f"Warning: CRD features requested but not explicitly extracted for {type(model).__name__}. Features will be None.")

    return logits, features


def _get_penultimate_dim(model_instance, device_to_use):
    """
    Dynamically determines the dimension of the feature vector before the final classifier.
    Caches the result in the model instance.
    """
    # Return cached dimension if it exists
    if hasattr(model_instance, 'penultimate_dim') and model_instance.penultimate_dim > 0:
        return model_instance.penultimate_dim

    # Ensure model is on the correct device and in eval mode for probing
    model_instance.to(device_to_use)
    original_mode_is_training = model_instance.training
    model_instance.eval()
    dim = 0

    with torch.no_grad():
        # Create a dummy input typical for CIFAR-10
        dummy_input = torch.randn(1, 3, 32, 32).to(device_to_use)
        _, features = get_model_output(model_instance, dummy_input, for_crd=True)
        if features is not None:
            dim = features.shape[1]
        else:
            print(f"Warning: _get_penultimate_dim: features were None for {type(model_instance).__name__}.")

    # Restore original mode
    if original_mode_is_training:
        model_instance.train()

    # Cache the dimension in the model instance for future use
    model_instance.penultimate_dim = dim

    if dim == 0:
        print(f"Warning: _get_penultimate_dim resulted in 0 for {type(model_instance).__name__}. CRD might fail.")

    return dim
