import tvm
from tvm import relay

import numpy as np
from tvm.contrib import graph_executor

from tvm.contrib.download import download_testdata

from tvm import relay, auto_scheduler
import tvm.relay.testing


# PyTorch imports
import torch
import torchvision

model_name = "mobilenet_v2"

model = getattr(torchvision.models, model_name)(pretrained=True).cuda()
model = model.eval()


input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape).cuda()
scripted_model = torch.jit.trace(model, input_data).eval()




target = tvm.target.Target("cuda")
input_name = "input0"
shape_list = [(input_name, input_shape)]

log_file = "mobilenet_v2.json"
dtype = "float32"

print("Extract tasks...")
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=50,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)
run_tuning()

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

