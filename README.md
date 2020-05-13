# dlib-pytorch-benchmark
A very naive and simple benchmark between dlib master and PyTorch 1.4.1 in terms of space and time.

This benchmarks were run on a NVIDIA GeForce GTX 1080 Ti with CUDA 10.2.89 and CUDNN 7.6.5.32 on Arch Linux.

## Model instantiation
Probably, this is a completely useless benchmark, but it's provided for completion, nonetheless.

### PyTorch
``` python
model = resnet50(pretrained=False)
```

### dlib
``` cpp
resnet<dlib::affine>::n50 net;
```

## 1st inference
This is also not very meaningful, since most of the time is spent allocating memory in the GPU.

### PyTorch
``` python
x = torch.zeros(32, 3, 224, 224)
x = x.cuda()
model = model.cuda()
# time measurement start
out = model(x)
# time measurement end
```

### dlib
``` cpp
dlib::matrix<dlib::rgb_pixel> image(224, 224);
dlib::assign_all_pixels(image, dlib::rgb_pixel(0, 0, 0));
std::vector<dlib::matrix<dlib::rgb_pixel>> minibatch(512, image);
```

At this point, we could just call:
``` cpp
const auto out = net(minibatch, 512);
```
But that wouldn't be a fair comparison, since it would do some extra work:
- apply softmax to the output of the net
- transfer the result from the device to the host

As a result, we need to forward a tensor that is already in the device.
There are several ways of doing it, here's one:

``` cpp
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

``` cpp
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

The first table shows the VRAM usage in MiB and the average timings in ms for different batch sizes for a tensor of shape Nx3x224x224.

|            | Memory  | (MiB)   |        | Time    | (ms)    |        |
|-----------:|--------:|--------:|-------:|--------:|--------:|-------:|
| batch size |    dlib | PyTorch | Factor | PyTorch |    dlib | Factor |
|          1 |     638 |     721 |  0.885 |   6.886 |  10.048 |  0.685 |
|          2 |     710 |     719 |  0.987 |   7.845 |  11.449 |  0.685 |
|          4 |     836 |     739 |  1.131 |  11.373 |  14.095 |  0.807 |
|          8 |    1074 |     775 |  1.386 |  17.504 |  19.303 |  0.907 |
|         16 |    1512 |     889 |  1.701 |  31.288 |  30.628 |  1.022 |
|         32 |    2510 |    1219 |  2.059 |  60.348 |  56.571 |  1.067 |
|         64 |    4342 |    1699 |  2.556 | 117.544 | 105.139 |  1.118 |
|        128 |    7976 |    2313 |  3.448 | 224.402 | 202.120 |  1.110 |

Results for the complete train cycle (transfer + forward + backward + loss + optimize):

|            | Memory |   (MiB) |        |    Time |    (ms) |        |
|-----------:|-------:|--------:|-------:|--------:|--------:|-------:|
| batch size |   dlib | PyTorch | Factor |    dlib | PyTorch | Factor |
|          1 |    973 |     991 |  0.982 |  39.292 |  47.571 |  0.826 |
|          2 |   1248 |    1119 |  1.115 |  29.308 |  51.219 |  0.572 |
|          4 |   1708 |    1281 |  1.333 |   40.95 |  60.329 |  0.679 |
|          8 |   2548 |    1645 |  1.549 |  65.193 |  78.995 |  0.825 |
|         16 |   4096 |    2389 |  1.715 | 113.596 | 116.117 |  0.978 |
|         32 |   7240 |    4061 |  1.783 | 218.968 | 203.942 |  1.074 |

## Conclusions

Regarding the inference time, dlib since to be substantially faster with small batch sizes (up to 8 samples) by taking between 25-40% less time than PyTorch.
As the batch size increases, the differences between both toolkits becomes minor.

As for the memory usage, PyTorch models are stateless, meaning that one can't access any intermediate values of the model.
On the dlib side, we can call `subnet()` on our `net` and then get the outputs, gradients (if we performed a backward pass), which makes it very easy to extract attention maps and perform grad-cam visualization.

However, I did observe that PyTorch memory peaks at 2929 MiB and 3843 MiB for batch sizes of 1 and 128 respectively. This is caused by the `torch.backends.cudnn.benchmark = True` setting.
