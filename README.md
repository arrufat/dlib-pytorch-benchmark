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
The times measured for each inference are around 6 ms, no matter the batch size (which is a good indicator that there are no memory transfers).

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
Here, the times measured for the first inference varies with the batch size (for 128 is around 90 ms).
However, the rest of forward calls are around 0.9 ms and indenpendent from the batch size.

Since the first call timing variability is systematic, we can just ignore it, since when the network works in a steady state the forward pass time is constant.

Nevertheless, if somebody has any idea of why this is happening, I would really love to know more.

## Results

This first table shows the results of the instantiation and first inference times for a tensor of shape 128x3x224x224.
As stated before, they are mostly meaningless:

| Test           |  PyTorch |   dlib   |
|---------------:|:--------:|:--------:|
|  instantiation |  239.672 |    0.078 |
|  1st inference | 1160.368 | 2609.590 |

The following table shows the VRAM usage in MiB and the average timings in ms for different batch sizes for a tensor of shape Nx3x224x224.

|            | Memory  | (MiB) |        | Time    | (ms)    |        |
|------------|---------|-------|--------|---------|---------|--------|
| batch size | PyTorch |  dlib | Factor | PyTorch | dlib    | Factor |
|          0 |     473 |   600 |   1.27 |     N/A |     N/A |    N/A |
|          1 |     711 |   632 |   0.89 |  12.581 |   7.647 |  0.608 |
|          2 |     709 |   706 |   0.99 |  14.060 |   8.448 |  0.601 |
|          4 |     729 |   840 |   1.15 |  16.850 |  12.088 |  0.717 |
|          8 |     765 |  1092 |   1.43 |  23.421 |  17.810 |  0.760 |
|         16 |     881 |  1556 |   1.77 |  34.879 |  30.440 |  0.873 |
|         32 |    1211 |  2604 |   2.15 |  60.421 |  58.028 |  0.960 |
|         64 |    1689 |  4536 |   2.68 | 110.507 | 112.568 |  1.019 |
|        128 |    2303 |  8374 |   3.64 | 214.652 | 222.337 |  1.036 |

## Conclusions

I am still not sure I am measuring the inference times in a fair way for both toolkits, so I will keep digging.

Regarding the inference time, dlib since to be substantially faster with small batch sizes (up to 8 samples) by taking between 25-30% less time than PyTorch.
As the batch size increases, the differences between both toolkits becomes minor.

As for the memory usage, PyTorch models are stateless, meaning that one can't access any intermediate values of the model.
On the dlib side, we can call `subnet()` on our `net` and then get the outputs, gradients (if we performed a backward pass), which makes it very easy to extract attention maps and perform grad-cam visualization.
