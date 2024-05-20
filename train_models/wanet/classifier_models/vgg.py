"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn


cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, norm_layer=nn.BatchNorm2d):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, norm_layer=nn.BatchNorm2d):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    norm_layer(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def vgg16(num_classes=10, norm_layer=nn.BatchNorm2d, **kwargs):
    model = VGG('VGG16', num_classes=num_classes, norm_layer=norm_layer)
    return model

