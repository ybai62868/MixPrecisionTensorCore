import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm import relay

# PyTorch imports
import torch
import torchvision

model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretained=True)
model = model.cuda()
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)


def get_nework(name, batch_size):
    input_shape = (batch_size, 3, 224, 224)
    input_data = torch.randn(input_shape)
    output_shape = (batch_size, 1000)
    input_name = "input0"
    shape_list = [(input_name, input_shape)]


    if "resnet" in name:
        model = getattr(torchvision.models, model_name)(pretained=True)
        model = model.cuda()
        model = model.eval()
        scripted_model = torch.jit.trace(model, input_data).eval()
        n_layer = int(name.split("_")[1])
        mod, params = relay.frontend.from_pytorch(scripted_model, input_shape)
    elif "mobile" in name:
        model = getattr(torchvision.models, model_name)(pretained=True)
        model = model.cuda()
        model = model.eval()
        scripted_model = torch.jit.trace(model, input_data).eval()
        n_layer = int(name.split("_")[1])
        mod, params = relay.frontend.from_pytorch(scripted_model, input_shape)
    elif "shuffle" in name:



    return mod, params, input_shape, output_shape