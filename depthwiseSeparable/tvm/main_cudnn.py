import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm import relay
from tvm.relay import testing
import tvm.testing

out_channels = 16
batch_size = 1

data = relay.var("data", relay.TensorType((batch_size, 3, 224, 224), "float32"))
weight = relay.var("weight")
bn_gamma = relay.var("bn_gamma")
bn_beta = relay.var("bn_beta")
bn_mmean = relay.var("bn_mean")
bn_mvar = relay.var("bn_var")

simple_net = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
)
simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
simple_net = relay.nn.relu(simple_net)
simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

data_shape = (batch_size, 3, 224, 224)
net, params = testing.create_workload(simple_net)


# import logging

# logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion

# target = "cuda"
# lib = relay.build_module.build(net, target, params=params)

# dev = tvm.device(target, 0)
# data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# module = runtime.GraphModule(lib["default"](dev))
# module.set_input("data", data)
# module.run()
# out_shape = (batch_size, out_channels, 224, 224)
# out = module.get_output(0, tvm.nd.empty(out_shape))
# out_cuda = out.numpy()


import torch
import torchvision

def get_network(model_name, batch_size, layout="NCHW", dtype="float32"):
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)
    
    input_shape = (batch_size, ) + image_shape
    output_shape = (batch_size, 1000)
    input_data = torch.randn(input_shape).cuda()
    input_name = "input0"
    shape_list = [(input_name, input_shape)]


    if "resnet" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.cuda()
        model = model.eval()
        scripted_model = torch.jit.trace(model, input_data).eval().cuda()
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif "mobile" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.cuda()
        model = model.eval()
        scripted_model = torch.jit.trace(model, input_data).eval().cuda()
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif "shuffle" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.cuda()
        model = model.eval()
        scripted_model = torch.jit.trace(model, input_data).eval().cuda()
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif "squeeze" in model_name:
        model = getattr(torchvision.models, model_name)(pretrained=True)
        model = model.cuda()
        model = model.eval()
        scripted_model = torch.jit.trace(model, input_data).eval().cuda()
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    else:
        raise ValueError("Unsupported network: " + name)


    return mod, params, input_shape, output_shape


# network = "mobilenet_v2"
network = "mobilenet_v2"
batch_size = 1
layout = "NCHW"
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)


# Extract tasks from the network
print("Extract tasks...")
net, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)


# net, params = testing.create_workload(simple_net)



target = "cuda -libs=cudnn"  # use cudnn for convolution
# target = "cuda"  # use cudnn for convolution
lib = relay.build_module.build(net, target, params=params)

dev = tvm.device(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = runtime.GraphModule(lib["default"](dev))
module.set_input("input0", data)

    

for _ in range(10):
    module.run()


# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=5, min_repeat_ms=500))

# import time
# avg_fwd_time = 0
# for i in range(100):
#     start = time.time()
#     module.run()
#     end = time.time()
#     fwd_time = end - start
#     avg_fwd_time += fwd_time
# avg_fwd_time = avg_fwd_time / 100
# print("Inference Time for cuDNN Backend: ", avg_fwd_time * 1000,"ms")
# out_shape = (batch_size, out_channels, 224, 224)
# out = module.get_output(0, tvm.nd.empty(out_shape))
# out_cudnn = out.numpy()

# tvm.testing.assert_allclose(out_cuda, out_cudnn, rtol=1e-5)
