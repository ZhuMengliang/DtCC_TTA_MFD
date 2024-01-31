#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
import torch.nn as nn


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class BasicBlock(nn.Module):
    # skip-connection
    expansion = 1

    def __init__(self, inplanes, cfg, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, cfg[0], stride)
        self.bn1 = nn.BatchNorm1d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(cfg[0], cfg[1])
        self.bn2 = nn.BatchNorm1d(cfg[1])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], in_channel=1, zero_init_residual=False, cfg=None):
        super(ResNet18, self).__init__()
        if cfg is None:
            cfg = [[64], [64, 64] * 2, [128, 128], [128, 128], [256, 256], [256, 256], [512, 512], [512, 512]]
            cfg = [item for sub_list in cfg for item in sub_list]
        n = 2
        self.inplanes = cfg[0]
        self.conv1 = nn.Conv1d(in_channel, out_channels=cfg[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(cfg[0])  # 这里也是要裁剪的
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], cfg=cfg[1:(2 * n + 1)])
        self.layer2 = self._make_layer(block, layers[1], cfg=cfg[(2 * n + 1):(4 * n + 1)], stride=2)
        self.layer3 = self._make_layer(block, layers[2], cfg=cfg[(4 * n + 1):(6 * n + 1)], stride=2)
        self.layer4 = self._make_layer(block, layers[3], cfg=cfg[(6 * n + 1):(8 * n + 1)], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.output_num = cfg[-1]
        self.cfg_model = cfg

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block=BasicBlock, blocks=2, cfg=None, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != cfg[1]:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, cfg[1], stride),
                nn.BatchNorm1d(cfg[1]),
            )

        layers = []
        layers.append(block(self.inplanes, cfg[0:2], stride, downsample))
        self.inplanes = cfg[1]
        for i in range(1, blocks):
            downsample = None
            stride = 1
            if stride != 1 or self.inplanes != cfg[1 + 2 * i]:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, cfg[1 + 2 * i], stride),
                    nn.BatchNorm1d(cfg[1 + 2 * i]),
                )
            layers.append(block(self.inplanes, cfg[2 * i: 2 * (i + 1)], downsample=downsample))
            self.inplanes = cfg[1 + 2 * i]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


# convnet without the last layer


if __name__ == '__main__':
    import torch

    model = ResNet18()
    x = torch.randn((10, 1, 30))  # batchsize,in_channels,length
    feature = model(x)

    for k, m in enumerate(model.modules()):
        print(m)
        if isinstance(m, nn.BatchNorm1d):
            print("hello")