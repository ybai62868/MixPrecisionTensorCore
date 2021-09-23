import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from nvstatsrecorder.recorders import NVStatsRecorder
torch.backends.cudnn.benchmark = True


batch_size = 1
ic = 3#256
ih = iw = 12#448
kh = kw = 5#3
stride = 1
pad = 0#1
dilation = 1
group = ic
oc = 256#16
oh = ow = 8#448

x = torch.rand(batch_size, ic, ih, iw).cuda()


def standard_conv():
    conv = nn.Conv2d(
    in_channels=ic, 
    out_channels=oc, 
    kernel_size=kw, 
    stride=stride, 
    padding=pad, 
    dilation=dilation,
    groups=1).cuda()
    FLOPs = batch_size * kh * kw * oc * ic * oh * ow
    print("MFLOPs in conv: ", FLOPs/1e6)
    out = conv(x)

    for _ in range(10):
        out = conv(x)
    
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
    print("Inference Time for conv: ", avg_fwd_time1*1000,"ms")
    print("TFLOPS in conv: ", FLOPs/1e12/avg_fwd_time1)
    
    param_mem = kw * kw * ic * oc
    feature_mem = oh * ow * oc 
    print("Total memory access in standard conv: ", param_mem + feature_mem)





# def depthwise_separable_conv:
def depthwise_sep():
    depth_conv = nn.Conv2d(
        in_channels=ic, 
        out_channels=ic, 
        kernel_size=kw, 
        stride=stride,
        padding=pad,
        dilation=dilation,
        groups=group,
        ).cuda()
    point_conv = nn.Conv2d(
        in_channels=ic, 
        out_channels=oc, 
        kernel_size=1)
    depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv).cuda()
    #params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)
    FLOPs = batch_size * (kw * kh * ic * ow * oh + ic * oc * ow * oh)
    print("MFLOPs in DSC: ", FLOPs/1e6)

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
    print("Inference Time for depthwise_sep: ", avg_fwd_time2*1000,"ms")
    print("TFLOPS in depthwise_sep: ", FLOPs/1e12/avg_fwd_time2)

    dw_param_mem = ic * kw * kh
    odw_feature_mem = ic * ow * oh
    pw_param_mem = ic * oc * 1 * 1
    feature_mem = oc * ow * oh
    print("Total memory access in depthwise_sep: ", dw_param_mem + odw_feature_mem + pw_param_mem + feature_mem)
    # print("Total memory access in depthwise_sep: ", dw_param_mem + pw_param_mem + feature_mem)


if __name__ == "__main__":
    standard_conv()
    depthwise_sep()
    