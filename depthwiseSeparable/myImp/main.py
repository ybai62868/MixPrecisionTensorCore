# Use python to implement digonal-wise matrix multiplication

import torch
import torch.nn as nn
import numpy as np
from models import *
import tvm
from tvm import relay
from tvm.contrib import graph_executor as runtime
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    net = alexnet(pretrained=False)
    net2 = mobilenet_v1(pretrained=False)
    net3 = mobilenet_v2(pretrained=True)
    model = net2.cuda()
    model = model.eval()


    batch_size = 1
    image_shape = (3, 224, 224)

    input_shape = (batch_size, ) + image_shape
    output_shape = (batch_size, 1000)
    input_data = torch.randn(input_shape).cuda()
    input_name = "input0"
    shape_list = [(input_name, input_shape)]



    scripted_model = torch.jit.trace(model, input_data).eval().cuda()
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    target = "cuda -libs=cudnn"
    # target = "cuda"  # use cudnn for convolution
    lib = relay.build_module.build(mod, target, params=params)

    dev = tvm.device(target, 0)
    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input("input0", data)

    
    # Warm-up
    for _ in range(10):
        module.run()


    # Evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=5, min_repeat_ms=500))