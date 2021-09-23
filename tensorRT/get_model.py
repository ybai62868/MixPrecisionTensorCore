import torch
import torchvision



model_name = "mobilenet_v2"
model = getattr(torchvision.models, model_name)(pretrained=False)
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape).cuda()
model = model.cuda()
model = model.eval()

onnx_file = "mobilenet_v2.onnx"
torch.onnx.export(
    model,
    input_data,
    onnx_file,
    opset_version = 11,
    input_names = ["input"],
    output_names = ["output"]
)
