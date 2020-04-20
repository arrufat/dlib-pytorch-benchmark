#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <dlib/image_processing.h>
#include <iostream>
#include <numeric>
#include <resnet.h>

namespace chrono = std::chrono;
using fms = chrono::duration<float, std::milli>;

int main(const int argc, const char** argv)try
{
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    chrono::time_point<chrono::steady_clock> t0, t1;
    float duration{};

    size_t mini_batch_size = 1;
    if (argc == 2)
    {
        mini_batch_size = std::stoul(argv[1]);
    }

    // Create the images to test
    dlib::matrix<dlib::rgb_pixel> image(224, 224);
    dlib::assign_all_pixels(image, dlib::rgb_pixel(0, 0, 0));
    std::vector<dlib::matrix<dlib::rgb_pixel>> minibatch(mini_batch_size, image);
    // Create some labels
    std::vector<unsigned long> labels(mini_batch_size, 0);

    // The input tensor of the network
    dlib::resizable_tensor x;

    {
        // Declare the network
        t0 = chrono::steady_clock::now();
        resnet::infer_50 net;
        t1 = chrono::steady_clock::now();
        duration = chrono::duration_cast<fms>(t1 - t0).count();
        std::cout << "instantiation time: " << duration << " ms" << std::endl;
        std::cin.get();

        // Convert them to a tensor for this network. This way we only measure inference and not
        // data transfer. In PyTorch, this would be: x = x.cuda()
        net.to_tensor(minibatch.begin(), minibatch.end(), x);
        t0 = chrono::steady_clock::now();
        net.forward(x);
        t1 = chrono::steady_clock::now();
        duration = chrono::duration_cast<fms>(t1 - t0).count();
        std::cout << "1st inference time: " << duration << " ms" << std::endl;
        std::cout << "input shape: " << x.num_samples() << 'x' << x.k() << 'x' << x.nr() << 'x'
                  << x.nc() << std::endl;
        std::cout << "parameters: " << dlib::count_parameters(net) << std::endl;
        std::cin.get();

        std::array<float, 100> times;
        for (auto& time : times)
        {
            net.to_tensor(minibatch.begin(), minibatch.end(), x);
            t0 = chrono::steady_clock::now();
            net.forward(x);
            t1 = chrono::steady_clock::now();
            duration = chrono::duration_cast<fms>(t1 - t0).count();
            std::cout << "2nd inference time: " << duration << " ms   \r" << std::flush;
            time = duration;
        }
        std::cout << "\navg: " << std::fixed << std::setprecision(3)
                  << std::accumulate(times.begin(), times.end(), 0.f, std::plus<float>()) /
                         times.size()
                  << " ms" << std::endl;
        std::cin.get();
        net.clean();
    }

    resnet::train_50 net;
    // measure backward pass
    std::vector<dlib::sgd> solvers(net.num_computational_layers, dlib::sgd(0.0005, 0.9));
    net.to_tensor(minibatch.begin(), minibatch.end(), x);
    net.compute_loss(x, labels.begin());
    net.back_propagate_error(x);
    net.update_parameters(solvers, 0.1);
    std::array<float, 100> times;
    for (auto& time : times)
    {
        t0 = chrono::steady_clock::now();
        net.to_tensor(minibatch.begin(), minibatch.end(), x);
        net.compute_loss(x, labels.begin());
        net.back_propagate_error(x);
        net.update_parameters(solvers, 0.1);
        t1 = chrono::steady_clock::now();
        duration = chrono::duration_cast<fms>(t1 - t0).count();
        std::cout << "backward pass time: " << duration << " ms   \r" << std::flush;
        time = duration;
    }
    std::cout << "\navg: " << std::fixed << std::setprecision(3)
              << std::accumulate(times.begin(), times.end(), 0.f, std::plus<float>()) /
                     times.size()
              << " ms" << std::endl;
    std::cin.get();

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
