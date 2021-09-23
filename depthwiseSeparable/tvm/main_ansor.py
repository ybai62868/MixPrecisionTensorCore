import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm import relay
from tvm.relay import Expr
from tvm.relay.analysis import free_vars
import pytest
# PyTorch imports
import torch
import torchvision

from tvm.contrib.download import download_testdata


# model_name = "resnet18"
# model = getattr(torchvision.models, model_name)(pretrained=True)
# model = model.cuda()
# model = model.eval()

# # We grab the TorchScripted model via tracing
# input_shape = [1, 3, 224, 224]
# input_data = torch.randn(input_shape)
# scripted_model = torch.jit.trace(model, input_data).eval()

# input_name = "input0"
# shape_list = [(input_name, input_shape)]
# mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

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
    else:
        raise ValueError("Unsupported network: " + name)


    return mod, params, input_shape, output_shape


network = "mobilenet_v2"
batch_size = 1
layout = "NCHW"
target = tvm.target.Target("cuda")
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)


# Extract tasks from the network
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
print(mod)

# import pdb;pdb.set_trace()
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)


for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

# run_tuning()


# Compile with the history best
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("input0", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))

