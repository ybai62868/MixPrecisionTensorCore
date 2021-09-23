import tvm
from tvm import relay
import time
import numpy as np
import torch
import torchvision

torch.backends.cudnn.benchmark = True
'''
'AlexNet', 'DenseNet', 'GoogLeNet', 'GoogLeNetOutputs', 
'Inception3', 'InceptionOutputs', 'MNASNet', 'MobileNetV2', 
'MobileNetV3', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'VGG', 
'_GoogLeNetOutputs', '_InceptionOutputs', '__builtins__', 
'__cached__', '__doc__', '__file__', '__loader__', '__name__', 
'__package__', '__path__', '__spec__', '_utils', 'alexnet', 'densenet', 
'densenet121', 'densenet161', 'densenet169', 'densenet201', 'detection', 
'googlenet', 'inception', 'inception_v3', 'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 
'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3_large',
 'mobilenet_v3_small', 'mobilenetv2', 'mobilenetv3', 'quantization', 'resnet', 
 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d',
  'resnext50_32x4d', 'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
   'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2', 'squeezenet', 'squeezenet1_0', 
   'squeezenet1_1', 'utils', 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'video', 'wide_resnet101_2', 'wide_resnet50_2']
'''

# model_name = "resnet18"
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=False)
model.half()
model = model.eval()
# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape).cuda()
model = model.cuda()
scripted_model = torch.jit.trace(model, input_data).eval()


# warm-up
for i in range(50):
    output = model(input_data)


avg_fwd_time = 0
for i in range(50):
    torch.cuda.synchronize()
    start = time.time()
    output = model(input_data)
    torch.cuda.synchronize()
    end = time.time()
    fwd_time = end - start
    avg_fwd_time += fwd_time
avg_fwd_time = avg_fwd_time * 1000 / 50
print("time cost for torch-cudnn-fp32: %.5f" % avg_fwd_time,"ms")


input_data = input_data.half()
model = model.half()
model.eval()

for i in range(50):
    output = model(input_data)

avg_fwd_time = 0
for i in range(50):
    torch.cuda.synchronize()
    start = time.time()
    output = model(input_data)
    torch.cuda.synchronize()
    end = time.time()
    fwd_time = end - start
    avg_fwd_time += fwd_time
avg_fwd_time = avg_fwd_time * 1000 / 50
print("time cost for torch-cudnn-fp16: %.5f" % avg_fwd_time,"ms")

import pdb;pdb.set_trace()

start = time.time()
for i in range(100):
    output = scripted_model(input_data)
    # output = model(input_data)


print("time cost for pytorch jit:", (time.time() - start) * 1000/100,"ms")

# # import pdb;pdb.set_trace()
# mod, params = relay.frontend.from_pytorch(scripted_model, [('input', input_shape)])

# # target = tvm.target.Target("llvm", host="llvm")
# dev = tvm.cuda(0)
# with tvm.transform.PassContext(opt_level=3):
#     # lib = relay.build(mod, target="cuda -libs=cudnn", params=params)
#     lib = relay.build(mod, target="cuda", params=params)



# from tvm.contrib import graph_executor
# dtype = "float32"
# m = graph_executor.GraphModule(lib["default"](dev))
# # Set inputs
# data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
# m.set_input('input', tvm.nd.array(data_tvm))
# # Execute
# for i in range(50):
#     m.run()
# start = time.time()
# for i in range(10):
#     m.run()
# end = time.time()
# print("time cost for tvm-cudnn: ", (end - start) * 1000/10,"ms")
# # Get outputs
# tvm_output = m.get_output(0)



