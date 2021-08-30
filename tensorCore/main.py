import torch
import torch.nn

torch.backends.cudnn.benchmark = True

batch_size, input, out = 256, 1024, 2048

tensor = torch.randn(batch_size, input).cuda().half()
layer = torch.nn.Linear(input, out).cuda().half()
layer(tensor)


