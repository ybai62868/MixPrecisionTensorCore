import torch
import torchvision
import numpy as np
from torch.cuda.amp import autocast as autocast

# ['AlexNet', 'DenseNet', 'GoogLeNet', 
# 'GoogLeNetOutputs', 'Inception3', 'InceptionOutputs', 
# 'MNASNet', 'MobileNetV2', 'ResNet', 
# 'ShuffleNetV2', 'SqueezeNet', 'VGG', 
# '_GoogLeNetOutputs', '_InceptionOutputs', '__builtins__', 
# '__cached__', '__doc__', '__file__', '__loader__', '__name__', 
# '__package__', '__path__', '__spec__', '_utils', 'alexnet', 
# 'densenet', 'densenet121', 'densenet161', 'densenet169', 
# 'densenet201', 'detection', 'googlenet', 'inception', 
# 'inception_v3', 'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 
# 'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 
# 'quantization', 'resnet', 'resnet101', 'resnet152', 'resnet18', 
# 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext50_32x4d', 
# 'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 
# 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2', 
# 'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 'utils', 
# 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 
# 'vgg16_bn', 'vgg19', 'vgg19_bn', 'video', 'wide_resnet101_2', 
# 'wide_resnet50_2']

model = torchvision.models.mobilenet_v2(pretrained=False)
device = torch.device("cuda")
model.to(device)
dummy_input = torch.randn(64, 3,224,224).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    # with autocast():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn,"ms")