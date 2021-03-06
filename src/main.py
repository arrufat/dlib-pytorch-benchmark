import torch
from os import environ
from sys import argv
from time import time
from torchvision.models import resnet50 as resnet

# torch.backends.cudnn.benchmark = True
environ["CUDA_LAUNCH_BLOCKING"] = "1"

minibatch_size = 1
if len(argv) > 1:
    minibatch_size = int(argv[1])

# if True:
with torch.no_grad():
    x = torch.zeros(minibatch_size, 3, 224, 224)
    print("input shape:", x.shape)

    t0 = time()
    model = resnet(pretrained=False).eval()
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
        model(x)
        t1 = time()
        times.append(t1 - t0)
        print("2nd inference time:", (t1 - t0) * 1000, "ms", end="\r")
    print()
    print("avg: {:.3f} ms".format(sum(times) * 1000 / len(times)))
    input()

# measure the backward pass
model = resnet(pretrained=False).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005
)

x = torch.zeros(minibatch_size, 3, 224, 224, requires_grad=True)
labels = torch.zeros((minibatch_size), dtype=torch.long)

# do one iteration outside of the loop for the memory allocation
optimizer.zero_grad()
outputs = model(x.cuda())
loss = criterion(outputs, labels.cuda())
loss.backward()
optimizer.step()

times = []
for epoch in range(100):
    x = x.cpu()
    labels = labels.cpu()
    t0 = time()
    optimizer.zero_grad()
    outputs = model(x.cuda())
    loss = criterion(outputs, labels.cuda())
    loss.backward()
    optimizer.step()
    # outputs = outputs.cpu()
    t1 = time()
    times.append(t1 - t0)
    print("backward pass time:", (t1 - t0) * 1000, "ms", end="\r")
print()
print("avg: {:.3f} ms".format(sum(times) * 1000 / len(times)))
input()
