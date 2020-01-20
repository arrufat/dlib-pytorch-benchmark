import torch
from os import environ
from sys import argv
from time import time
from torchvision.models import resnet50 as resnet

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
    model = resnet(pretrained=False)
    t1 = time()
    print("instantiation time:", (t1 - t0) * 1000, "ms")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:", num_params)
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
        t1 = time()
        times.append(t1 - t0)
        print("2nd inference time:", (t1 - t0) * 1000, "ms", end='\r')
    print()
    print("avg: {:.3f} ms".format(sum(times) * 1000 / len(times)))
    input()

# measure the backward pass
model = resnet(pretrained=False)
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

x = torch.zeros(minibatch_size, 3, 224, 224, requires_grad=True)
model(x.cuda())
labels = torch.zeros((minibatch_size), dtype=torch.long)
times = []
for epoch in range(100):
    x = x.cpu()
    labels = labels.cpu()
    t0 = time()
    x = x.cuda()
    labels = labels.cuda()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    t1 = time()
    times.append(t1 - t0)
    print("backward pass time:", (t1 - t0) * 1000, "ms", end='\r')
print()
print("avg: {:.3f} ms".format(sum(times) * 1000 / len(times)))
input()
