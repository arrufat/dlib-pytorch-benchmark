#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <iostream>

#include <dlib/image_processing.h>

#include "resnet.h"

using fms = std::chrono::duration<float, std::milli>;

int main(const int argc, const char** argv) try
{
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    std::chrono::time_point<std::chrono::steady_clock> t0, t1;
    float duration{};

    size_t mini_batch_size = 1;

    if (argc == 2)
        mini_batch_size = std::stoul(argv[1]);

    // dlib::set_dnn_prefer_smallest_algorithms();
    // Declare the network
    t0 = std::chrono::steady_clock::now();
    resnet<dlib::affine>::l50 net;
    t1 = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<fms>(t1 - t0).count();
    std::cout << "instantiation time: " << duration << " ms" << std::endl;
    std::cin.get();

    // Create the images to test
    dlib::matrix<dlib::rgb_pixel> image(224, 224);
    dlib::assign_all_pixels(image, dlib::rgb_pixel(0, 0, 0));
    std::vector<dlib::matrix<dlib::rgb_pixel>> minibatch(mini_batch_size, image);
    // create some labels

    // Convert them to a tensor for this network. This way we only measure inference and not data transfer
    // In PyTorch, this would be: x = x.cuda()
    dlib::resizable_tensor x;
    net.to_tensor(minibatch.begin(), minibatch.end(), x);
    x.host();
    x.device();
    t0 = std::chrono::steady_clock::now();
    net.subnet().forward(x);
    t1 = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<fms>(t1 - t0).count();
    std::cout << "1st inference time: " << duration << " ms" << std::endl;
    std::cout << "input shape: " << x.num_samples() << 'x' << x.k() << 'x' << x.nr() << 'x' << x.nc() << std::endl;
    std::cout << "parameters: " << dlib::count_parameters(net) << std::endl;
    std::cin.get();

    std::array<float, 100> times;
    for (auto& time : times)
    {
        x.host();
        x.device();
        t0 = std::chrono::steady_clock::now();
        net.subnet().forward(x);
        t1 = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<fms>(t1 - t0).count();
        std::cout << "2nd inference time: " << duration << " ms   \r" << std::flush;
        time = duration;
    }
    std::cout << std::endl;
    std::cout <<
        "avg: " << std::fixed << std::setprecision(3) <<
        std::accumulate(times.begin(), times.end(), 0.f, std::plus<float>()) / times.size() <<
        " ms" << std::endl;
    std::cin.get();

    return EXIT_SUCCESS;
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
