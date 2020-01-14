import torch
from sys import argv
from time import time
from torchvision.models import resnet50

torch.backends.cudnn.benchmark = True

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

    for i in range(10):
        x = x.cpu().cuda()
        t0 = time()
        out = model(x)
        # out = out.cpu()
        t1 = time()
        print("2nd inference time:", (t1 - t0) * 1000, "ms")
    input()

    # model2 = resnet50(pretrained=True).cuda()
    # out = model2(x.clone())
    # print("2nd model loaded")
    # input()
