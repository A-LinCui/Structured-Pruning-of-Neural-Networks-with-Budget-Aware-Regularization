from typing import Union, List

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from wrapper import BARStructuredWrapper


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def add_wrapper(self, before_bn: bool = False, **kwargs) -> None:
        if before_bn:
            self.conv1 = BARStructuredWrapper(self.conv1, **kwargs)
            self.conv2 = BARStructuredWrapper(self.conv2, **kwargs)
            if len(self.shortcut) > 0:
                self.shortcut[0] = BARStructuredWrapper(self.shortcut[0], **kwargs)
        else:
            self.bn1 = BARStructuredWrapper(self.bn1, **kwargs)
            self.bn2 = BARStructuredWrapper(self.bn2, **kwargs)
            if len(self.shortcut) > 0:
                self.shortcut[1] = BARStructuredWrapper(self.shortcut[1], **kwargs)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def add_wrapper(self, before_bn: bool = False, **kwargs) -> None:
        if before_bn:
            self.conv1 = BARStructuredWrapper(self.conv1, **kwargs)
            self.conv2 = BARStructuredWrapper(self.conv2, **kwargs)
            self.conv3 = BARStructuredWrapper(self.conv3, **kwargs)
            if len(self.shortcut) > 0:
                self.shortcut[0] = BARStructuredWrapper(self.shortcut[0], **kwargs)
        else:
            self.bn1 = BARStructuredWrapper(self.bn1, **kwargs)
            self.bn2 = BARStructuredWrapper(self.bn2, **kwargs)
            self.bn3 = BARStructuredWrapper(self.bn3, **kwargs)
            if len(self.shortcut) > 0:
                self.shortcut[1] = BARStructuredWrapper(self.shortcut[1], **kwargs)


class CIFARResNet(nn.Module):
    def __init__(self, block: Union[Bottleneck, BasicBlock], 
            num_blocks: List[int], num_classes: int = 10) -> None:
        super(CIFARResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Union[Bottleneck, BasicBlock], 
            planes: int, num_blocks: int, stride: int) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def add_wrapper(self, before_bn: bool = False, **kwargs) -> None:
        def layer_add_wrapper(layer: nn.Module, before_bn: bool = False, **kwargs) -> None:
            for block in layer:
                block.add_wrapper(before_bn, **kwargs)
        
        if before_bn:
            self.conv1 = BARStructuredWrapper(self.conv1, **kwargs)
        else:
            self.bn1 = BARStructuredWrapper(self.bn1, **kwargs)
        layer_add_wrapper(self.layer1, before_bn, **kwargs)
        layer_add_wrapper(self.layer2, before_bn, **kwargs)
        layer_add_wrapper(self.layer3, before_bn, **kwargs)
        layer_add_wrapper(self.layer4, before_bn, **kwargs)


def CIFARResNet18(num_classes: int) -> CIFARResNet:
    return CIFARResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def CIFARResNet34(num_classes: int) -> CIFARResNet:
    return CIFARResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def CIFARResNet50(num_classes: int) -> CIFARResNet:
    return CIFARResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def CIFARResNet101(num_classes: int) -> CIFARResNet:
    return CIFARResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def CIFARResNet152(num_classes: int) -> CIFARResNet:
    return CIFARResNet(Bottleneck, [3, 8, 36, 3], num_classes)
