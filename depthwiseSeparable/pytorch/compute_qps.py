import torch
import torchvision
from torchsummary import summary
from thop import profile

# 2080 ti: 
# Memory Bandwidth 616GB/s
# peak FP32 14.2 TFLOPS
# peak FP16 28.5 TFLOPS
# Intensity-FP32 23
# Intensity-FP16 46.2

optimal_batch_size = 114
model = torchvision.models.mobilenet_v2(pretrained=False)
device = torch.device("cuda")
model.to(device)
# summary(model, (3, 224, 224))

# import pdb;pdb.set_trace()
dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)
flops, params = profile(model, inputs=(dummy_input, ))
print(flops, params)
params_mem = params * 4 / 1024 / 1024
print("MFLOPs of network: ", flops/1e6)
print("Memory of network: ", params_mem)
print("Intensity: ", flops/params_mem)
import pdb;pdb.set_trace()


repetitions=100
total_time = 0
with torch.no_grad():
  for rep in range(repetitions):
     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
     starter.record()
     _ = model(dummy_input)
     ender.record()
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)/1000
     total_time += curr_time
Throughput = (repetitions*optimal_batch_size)/total_time
print("Final Throughput: ",Throughput,"qps")