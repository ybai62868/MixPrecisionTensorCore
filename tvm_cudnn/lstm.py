from __future__ import print_function
import argparse
import time

import torch
from torch.autograd import Variable

torch.backends.cudnn.benchmark = True

def update_progress(progress):
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50),
                                                  progress * 100), end="")


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seconds', type=int, default=15)
parser.add_argument('--dry_runs', type=int, default=50)
parser.add_argument('--runs', type=int, default=50)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--hidden_size', default=640, type=int)
parser.add_argument('--half', action='store_true', dest='half')

args = parser.parse_args()
hidden_size = args.hidden_size
input = Variable(torch.randn(750, args.batch_size,
                    hidden_size).cuda())  # seq_length based on max deepspeech length 15 seconds

model = torch.nn.LSTM(hidden_size, hidden_size, num_layers=args.num_layers).cuda()

if args.half:
    input = input.half()
    model = model.half()

model.eval()


def run_benchmark():
    for n in range(args.dry_runs):
        output, (hx, cx) = model(input)
        # grad = output.data.clone().normal_()
        # output.backward(grad)
        update_progress(n / (float(args.dry_runs) - 1))
    print('\nDry runs finished, running benchmark')
    avg_fwd_time = 0
    torch.cuda.synchronize()
    for n in range(args.runs):
        torch.cuda.synchronize()
        start = time.time()
        output, (hx, cx) = model(input)
        torch.cuda.synchronize()
        end = time.time()
        fwd_time = end - start
        avg_fwd_time += fwd_time

    return avg_fwd_time * 1000 / float(args.runs)


if args.half:
    print("Running half precision benchmark")
else:
    print("Running standard benchmark")
avg_fwd_time = run_benchmark()

print('\n')
print("Avg Forward time: %.2fms " % avg_fwd_time)