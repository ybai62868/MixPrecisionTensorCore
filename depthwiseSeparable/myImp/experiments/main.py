# Use python to implement digonal-wise matrix multiplication

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

if __name__ == "__main__":
    # net = alexnet(pretrained=False)
    # net2 = mobilenet_v1(pretrained=False)
    # net3 = mobilenet_v2(pretrained=True)
    # net4 = nasnetamobile(pretrained='imagenet')
    # net5 = nasnetalarge(num_classes=1001, pretrained='imagenet+background')
    # 'nasnetalarge': 331,


    net3 = shufflenet_v2_x0_5(pretrained=False)
    model = net3.cuda()
    model.eval()

    scripted_model = torch.jit.trace(model, input_data).eval().cuda()
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # print(mod) 
    # import pdb;pdb.set_trace()

    target = "cuda -libs=cudnn"
    # target = "cuda"  # use cudnn for convolution
    lib = relay.build_module.build(mod, target, params=params)

    dev = tvm.device(target, 0)
    data = np.random.uniform(-1, 1, size=input_shape).astype("float16")
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input("input0", data)

    
    # Warm-up
    for _ in range(10):
        module.run()


    # Evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=5, min_repeat_ms=500))


