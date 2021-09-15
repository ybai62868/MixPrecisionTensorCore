import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nvstatsrecorder.recorders import NVStatsRecorder
torch.backends.cudnn.benchmark = False

batch_size = 1
height = 224
width = 224
out_channel = 128


def standard_conv(num, x):
    conv = nn.Conv2d(in_channels=num, out_channels=out_channel, kernel_size=3).cuda().half()
    # params = sum(p.numel() for p in conv.parameters() if p.requires_grad)
    FLOPs = 3 * 3 * out_channel * num * height * width
    out = conv(x)

    for i in range(10):
        output = conv(x)
    
    avg_fwd_time1 = 0
    for i in range(100):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            ouput = conv(x)
        torch.cuda.synchronize()
        end = time.time()
        fwd_time = end - start
        avg_fwd_time1 += fwd_time
    avg_fwd_time1= avg_fwd_time1 / 100
    print("TFLOPS in conv: ", batch_size*FLOPs/1e12/avg_fwd_time1)
    return 1000/avg_fwd_time1


def depthwise_sep(num, x):
    depth_conv = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=3, groups=num).cuda().half()
    point_conv = nn.Conv2d(in_channels=num, out_channels=out_channel, kernel_size=1).cuda().half()
    depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv).cuda().half()
    #params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)
    FLOPs = 3 * 3 * num * height * width + num * out_channel * height * width
    out_depthwise = depthwise_separable_conv(x)

    for i in range(10):
        ouput = depthwise_separable_conv(x)
    
    avg_fwd_time2 = 0
    for i in range(100):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            ouput = depthwise_separable_conv(x)
        torch.cuda.synchronize()
        end = time.time()
        fwd_time = end - start
        avg_fwd_time2 += fwd_time
    avg_fwd_time2 = avg_fwd_time2 / 100   
    print("TFLOPS in depthwise_sep: ", batch_size * FLOPs/1e12/avg_fwd_time2)

    return 1000/avg_fwd_time2



if __name__ == "__main__":
    channel_list = [pow(2, x) for x in range(3, 8)]
    latency_list_conv = []
    latency_list_convsep = []
    for num in channel_list: 
        in_channel = num * out_channel
        # print(in_channel)
        x = torch.rand(batch_size, in_channel, height, width).cuda().half()
        res1 = standard_conv(in_channel, x)
        res2 = depthwise_sep(in_channel, x)
        latency_list_conv.append(res1)
        latency_list_convsep.append(res2)
    # print(latency_list_conv, latency_list_convsep)
    plt.plot(channel_list, latency_list_conv, label = "standard conv")
    plt.plot(channel_list, latency_list_convsep, label = "depthwise separable conv")
    plt.xlabel("input/output channels")
    plt.ylabel("throughput")
    plt.title("pytorch benchmark")
    plt.legend()
    plt.savefig("./res.jpg")

# depth_conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, groups=128).cuda()
# point_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1).cuda()

# depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv).cuda()
# params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)

# out_depthwise = depthwise_separable_conv(x)

# for i in range(20):
#     ouput = conv(x)

# avg_fwd_time2 = 0
# for i in range(100):
#     torch.cuda.synchronize()
#     start = time.time()
#     with torch.no_grad():
#         ouput = depthwise_separable_conv(x)
#     torch.cuda.synchronize()
#     end = time.time()
#     fwd_time = end - start
#     avg_fwd_time2 += fwd_time
# avg_fwd_time2 = avg_fwd_time2 * 1000 / 100
# print(f"Time cost depthwise separable conv for {avg_fwd_time2} ms.")


# print(f"The standard convolution uses {params} parameters.")
# print(f"The depthwise separable convolution uses {params_depthwise} parameters.")


# print(f"The speed up between depthwise separable conv and standard conv {avg_fwd_time1/avg_fwd_time2}")
# print(f"The parameters times between depthwise separable conv and standard conv {params/params_depthwise}")

# assert out.shape == out_depthwise.shape, "Size mismatch"