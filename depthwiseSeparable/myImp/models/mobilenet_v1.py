import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MobileNetV1", "mobilenet_v1"]

class Block(nn.Module):
    '''
    Depthwise Conv
    Pointwise Conv
    '''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        # DW
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, 
            out_channels=in_planes, 
            kernel_size=3, 
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)

        # PW 
        self.conv2 = nn.Conv2d(
            in_channels=in_planes, 
            out_channels=out_planes, 
            kernel_size=1, 
            stride=1,
            padding=0,
            groups=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("Output Feature Map of DW: ", out.size())
        out = F.relu(self.bn2(self.conv2(out)))
        # print("Output Feature Map of PW: ", out.size())
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    depth_mul = 1
    cfg = [64, 
          (128,2), 
          128, 
          (256,2), 
          256, 
          (512,2), 
          512, 
          512, 
          512, 
          512, 
          512, 
          (1024,2), 
          1024]

    def __init__(self, block, num_classes):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(block, in_planes=32)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(int(1024 * self.depth_mul), num_classes)

    # Depthwise Separable Conv
    def _make_layers(self, block, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = int(x * self.depth_mul) if isinstance(x, int) else int(x[0]*self.depth_mul)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


def MobileNetV1(num_classes=1000):
    return MobileNet(Block, num_classes)


def test():
    net = MobileNetV1()
    x = torch.randn((1, 3, 224, 224))
    y = net(x)
    print(y.size())

def mobilenet_v1(pretrained: bool = False, progress: bool = True) -> MobileNet:
    model = MobileNetV1()
    return model



# if __name__ == "__main__":
#     test()