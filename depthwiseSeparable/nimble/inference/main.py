import torch
import torchvision


batch_size = 1

model = torchvision.models.resnet50()
model = model.cuda()
model.eval()

input_shape = [batch_size, 3, 224, 224]
dummy_input = torch.randn(*input_shape).cuda()


nimble_model = torch.cuda.Nimble(model)
nimble_model.prepare(dummy_input, training=False)

rand_input = torch.rand(*input_shape).cuda()
output = nimble_model(rand_input)

