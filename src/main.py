import torch
from os import environ
from sys import argv
from time import time
from torchvision.models import resnet50

torch.backends.cudnn.benchmark = True
environ["CUDA_LAUNCH_BLOCKING"] = "1"

minibatch_size = 1
if len(argv) > 1:
    minibatch_size = int(argv[1])

# if True:
with torch.no_grad():
    x = torch.zeros(minibatch_size, 3, 224, 224)
    print('input shape:', x.shape)

    t0 = time()
    model = resnet50(pretrained=False)
    t1 = time()
    print("instantiation time:", (t1 - t0) * 1000, "ms")
    input()

    t0 = time()
    model = model.cuda()
    x = x.cuda()
    model(x)
    t1 = time()
    print("1st inference time:", (t1 - t0) * 1000, "ms")
    input()

    times = []
    for i in range(100):
        x = x.cpu().cuda()
        t0 = time()
        out = model(x)
        out = out.cpu()
        t1 = time()
        times.append(t1 - t0)
        print("2nd inference time:", (t1 - t0) * 1000, "ms", end='\r')
    print()
    print("avg: {:.3f} ms".format(sum(times) * 1000 / len(times)))
    input()
