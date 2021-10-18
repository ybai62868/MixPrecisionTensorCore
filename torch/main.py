import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
import torchvision
import torchvision.models as models
import os
import numpy as np
from torch.cuda.amp import autocast as autocast

'''
'AlexNet', 'DenseNet', 'GoogLeNet', 'GoogLeNetOutputs', 'Inception3', 
'InceptionOutputs', 'MNASNet', 'MobileNetV2', 'ResNet', 'ShuffleNetV2', 
'SqueezeNet', 'VGG', '_GoogLeNetOutputs', '_InceptionOutputs', '__builtins__',
'__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', 
'__path__', '__spec__', '_utils', 'alexnet', 'densenet', 'densenet121',
 'densenet161', 'densenet169', 'densenet201', 'detection', 'googlenet', 
 'inception', 'inception_v3', 'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 
 'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'quantization', 
 'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 
 'resnext101_32x8d', 'resnext50_32x4d', 'segmentation', 'shufflenet_v2_x0_5', 
 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2',
  'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 'utils', 'vgg', 'vgg11', 
  'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
   'video', 'wide_resnet101_2', 'wide_resnet50_2'
'''
# model_name = "resnet18"
model_name = "mobilenet_v2"
model = getattr(torchvision.models, model_name)(pretrained=False)

# model = models.mobilenet(pretrained=False)
model.half()
# We grab the TorchScripted model via tracing
input_shape = [128, 3, 224, 224]
input_data = torch.randn(input_shape).cuda().half()
model = model.cuda()
model = model.eval()


# print(model)


# warm-up
for i in range(20):
    output = model(input_data)

avg_fwd_time = 0
for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        # with autocast():
            output = model(input_data)
    torch.cuda.synchronize()
    end = time.time()
    fwd_time = end - start
    avg_fwd_time += fwd_time
avg_fwd_time = avg_fwd_time * 1000 / 100
print("time cost for : %.5f" % avg_fwd_time,"ms")



