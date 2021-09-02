import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
from mxnet.gluon.model_zoo.vision import get_model
import mxnet

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

# shape_dict = {"data": data_shape}
# block = get_model("resnet18_v1", pretrained=True)
# mod, params = relay.frontend.from_mxnet(block, shape_dict, dtype = "float16")

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape,
    dtype="float32"
)

# mod, params = relay.testing.mobilenet.get_workload(
#     batch_size=batch_size, image_shape=image_shape,
#     dtype="float16"
# )


# multiplier = 1
# block = mxnet.gluon.model_zoo.vision.get_mobilenet_v2(multiplier, pretrained=True)
# mod, params = relay.frontend.from_mxnet(
#             block, 
#             shape={"data": data_shape}, 
#             dtype="float16"
# )

# block = mxnet.gluon.model_zoo.vision.get_resnet(1, 18, pretrained=True)
# mod, params = relay.frontend.from_mxnet(
#             block, 
#             shape={"data": data_shape}, 
#             dtype="float32"
# )



opt_level = 3
target = "cuda -libs=cudnn"
# target = "cuda"

with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

dev = tvm.device(target, 0)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("data", data)
module.run()

import timeit
timing_number = 10
timing_repeat = 10
optimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}
print("optimized: %s ms" % (optimized))