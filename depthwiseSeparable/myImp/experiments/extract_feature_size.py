
import torch
import torch.nn as nn
import onnx
import numpy as np
from models import *
import tvm
from tvm import relay
from tvm.contrib import graph_executor as runtime
torch.backends.cudnn.benchmark = True


batch_size = 1
image_shape = (3, 224, 224)
input_shape = (batch_size, ) + image_shape
input_data = torch.randn(input_shape).cuda()
input_name = "input0"
shape_list = [(input_name, input_shape)]

'''
DW: [M, M*K*K] X [M*K*K, F*F]
PW: [N, M] X [M, F*F]

'''

if __name__ == "__main__":
    model = mobilenet_v1(pretrained=False)
    # print(model)