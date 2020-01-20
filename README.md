# dlib-pytorch-benchmark
A very naive and simple benchmark between dlib and pytorch in terms of space and time.

This benchmarks were run on a NVIDIA GeForce GTX 1080 Ti with CUDA 10.2.89 and CUDNN 7.6.5.32 on Arch Linux.

## Model instantiation
Probably, this is a completely useless benchmark, but it's provided for completion, nonetheless.

### PyTorch
``` python
model = resnet50(pretrained=False)
```

### dlib
``` c++
resnet<dlib::affine>::_50 net;
```

## 1st inference
This is also not very meaningful, since most of the time is spent allocating memory in the GPU.

### PyTorch
``` python
x = torch.zeros(512, 3, 224, 224)
x = x.cuda()
model = model.cuda()
# time measurement start
out = model(x)
# time measurement end
```

### dlib
``` c++
dlib::matrix<dlib::rgb_pixel> image(224, 224);
dlib::assign_all_pixels(image, dlib::rgb_pixel(0, 0, 0));
std::vector<dlib::matrix<dlib::rgb_pixel>> minibatch(512, image);
```

At this point, we could just call:
``` c++
const auto out = net(minibatch, 512);
```
But that wouldn't be a fair comparison, since it would do some extra work:
- apply softmax to the output of the net
- transfer the result from the device to the host

As a result, we need to forward a tensor that is already in the device.
There are several ways of doing it, here's one:

``` c++
dlib::resizable tensor x;
net.to_tensor(minibatch.begin(), minibatch.end(), x);
x.device();
// time measurement start
net.subnet().forward(x);
// time measurement end
```
Now dlib is doing exactly the same operations as PyTorch, as far as I know.

## Next inferences
In my opininion, the most important benchmark is this one.
It measures how the network performs once it has been "warmed up".

For this part, I decided not to count the cuda syncronization time, only the inference time for a tensor that is already in the device.

### PyTorch
In PyTorch, every time I forward the network, I make sure all the transfers between the host and the device have been finished:

``` python
for i in range(10):
    x = x.cpu().cuda()
    # time measurement start
    out = model(x)
    # time measurement end
```

### dlib
For dlib I followed a similar pattern:

``` c++
for (int i = 0; i < 10; ++i)
{
    x.host();
    x.device();
    // time measurement start
    net.subnet().forward(x);
    // time measurement end
}
```

## Results

This first table shows the results of the instantiation and first inference times for a tensor of shape 128x3x224x224.
As stated before, they are mostly meaningless:

| Test           |  PyTorch |   dlib   |
|---------------:|:--------:|:--------:|
|  instantiation |  239.672 |    0.078 |
|  1st inference | 1160.368 | 2609.590 |

The following table shows the VRAM usage in MiB and the average timings in ms for different batch sizes for a tensor of shape Nx3x224x224.

|            | Memory  | (MiB) |        | Time    | (ms)    |        |
|-----------:|--------:|------:|-------:|--------:|--------:|-------:|
| batch size | PyTorch |  dlib | Factor | PyTorch |    dlib | Factor |
|          1 |     691 |   640 |  0.915 |  12.581 |   7.647 |  0.608 |
|          2 |     689 |   716 |  1.039 |  14.060 |   8.448 |  0.601 |
|          4 |     707 |   838 |  1.185 |  16.850 |  12.088 |  0.717 |
|          8 |     759 |  1076 |  1.418 |  23.421 |  17.810 |  0.760 |
|         16 |     881 |  1506 |  1.701 |  34.879 |  30.440 |  0.873 |
|         32 |    1029 |  2504 |  2.433 |  60.421 |  58.028 |  0.960 |
|         64 |    1555 |  4336 |  2.788 | 110.507 | 112.568 |  1.019 |
|        128 |    2411 |  7970 |  3.301 | 214.652 | 220.621 |  1.028 |

## Conclusions

Regarding the inference time, dlib since to be substantially faster with small batch sizes (up to 8 samples) by taking between 25-40% less time than PyTorch.
As the batch size increases, the differences between both toolkits becomes minor.

As for the memory usage, PyTorch models are stateless, meaning that one can't access any intermediate values of the model.
On the dlib side, we can call `subnet()` on our `net` and then get the outputs, gradients (if we performed a backward pass), which makes it very easy to extract attention maps and perform grad-cam visualization.

However, I did observe that PyTorch memory peaks at 2929 MiB and 3843 MiB for batch sizes of 1 and 128 respectively. This is caused by the `torch.backends.cudnn.benchmark = True` setting.
