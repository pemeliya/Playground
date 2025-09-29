// hipcc -std=c++17 sync_test.cpp -o sync_test -lpthread
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>
#include <csignal>
#include <pthread.h>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = call;                                                 \
        if (err != hipSuccess) {                                               \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << err << " (" << hipGetErrorString(err)    \
                      << ")\n";                                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static std::atomic<bool> g_stop{false};

void sigint_handler(int) {
    g_stop.store(true);
}

// Pin a std::thread to a specific CPU core (Linux only, best-effort)
void set_thread_affinity(std::thread &t, unsigned core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t handle = t.native_handle();
    pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset);
}

// Simple device kernel
__global__ void add_kernel(float *d_out, const float *d_in, float v, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_out[idx] = d_in[idx] + v;
}

int main() {
    std::signal(SIGINT, sigint_handler);

    unsigned hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;
    std::cout << "Detected hardware_concurrency(): " << hw_threads << " threads\n";

    // Launch CPU worker threads
    std::vector<std::thread> workers;
    std::atomic<size_t> copy_ops{0};

    for (unsigned i = 0; i < hw_threads; ++i) {
        workers.emplace_back([i, &copy_ops]() {
            const size_t BUF_BYTES = 4 * 1024 * 1024; // 4 MB
            std::vector<char> a(BUF_BYTES), b(BUF_BYTES);
            std::memset(a.data(), (int)i, BUF_BYTES);
            while (!g_stop.load()) {
                std::memcpy(b.data(), a.data(), BUF_BYTES);
                volatile char c = b[0]; (void)c;
                ++copy_ops;
            }
        });
        set_thread_affinity(workers.back(), i);
    }

    // --- GPU setup ---
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    std::cout << "hipGetDeviceCount() = " << deviceCount << "\n";

    int useDevices = std::min(deviceCount, 4);
    if (useDevices <= 0) {
        std::cerr << "No HIP devices found.\n";
        g_stop.store(true);
    }

    struct PerDevice {
        int devId;
        hipDeviceProp_t props;
        std::vector<hipStream_t> streams;
        std::vector<void*> d_inputs;
        std::vector<void*> d_outputs;
        std::vector<void*> h_pinned_src;
        std::vector<void*> h_pinned_dst;
    };
    std::vector<PerDevice> devices;

    const size_t ELEMENTS = 1 << 20; // 1M floats (~4MB)
    const size_t BYTES = ELEMENTS * sizeof(float);
    const int streamsPerDevice = 4;

    for (int d = 0; d < useDevices; ++d) {
        PerDevice pd;
        pd.devId = d;
        HIP_CHECK(hipGetDeviceProperties(&pd.props, d));
        std::cout << "Device " << d << ": " << pd.props.name << "\n";

        HIP_CHECK(hipSetDevice(d));

        pd.streams.resize(streamsPerDevice);
        pd.d_inputs.resize(streamsPerDevice);
        pd.d_outputs.resize(streamsPerDevice);
        pd.h_pinned_src.resize(streamsPerDevice);
        pd.h_pinned_dst.resize(streamsPerDevice);

        for (int s = 0; s < streamsPerDevice; ++s) {
            HIP_CHECK(hipStreamCreateWithFlags(&pd.streams[s], hipStreamNonBlocking));
            HIP_CHECK(hipMalloc(&pd.d_inputs[s], BYTES));
            HIP_CHECK(hipMalloc(&pd.d_outputs[s], BYTES));
            HIP_CHECK(hipHostMalloc(&pd.h_pinned_src[s], BYTES, hipHostMallocDefault));
            HIP_CHECK(hipHostMalloc(&pd.h_pinned_dst[s], BYTES, hipHostMallocDefault));

            float *hsrc = static_cast<float*>(pd.h_pinned_src[s]);
            for (size_t i = 0; i < ELEMENTS; ++i) hsrc[i] = static_cast<float>(i % 1024);
        }
        devices.push_back(std::move(pd));
    }

    std::cout << "Running... press Ctrl+C to stop.\n";
    size_t iteration = 0;

    while (!g_stop.load()) {
        for (auto &pd : devices) {
            HIP_CHECK(hipSetDevice(pd.devId));
            for (size_t s = 0; s < pd.streams.size(); ++s) {
                hipStream_t stream = pd.streams[s];

                HIP_CHECK(hipMemcpyHtoDAsync(pd.d_inputs[s], pd.h_pinned_src[s], BYTES, stream));

                int block = 256;
                int grid = (ELEMENTS + block - 1) / block;
                hipLaunchKernelGGL(add_kernel,
                                   dim3(grid), dim3(block), 0, stream,
                                   static_cast<float*>(pd.d_outputs[s]),
                                   static_cast<float*>(pd.d_inputs[s]),
                                   1.2345f, ELEMENTS);

                HIP_CHECK(hipMemcpyDtoHAsync(pd.h_pinned_dst[s], pd.d_outputs[s], BYTES, stream));
            }
        }

        // synchronize all devices
        for (auto &pd : devices) {
            HIP_CHECK(hipSetDevice(pd.devId));
            HIP_CHECK(hipDeviceSynchronize());
        }

        // print after all synchronizations
        std::cout << "Iteration " << iteration++
                  << " done, devices=" << devices.size()
                  << ", streams/device=" << (devices.empty() ? 0 : devices[0].streams.size())
                  << ", hipMemcpyHtoDAsync() + hipLaunchKernelGGL() + hipMemcpyDtoHAsync() + hipDeviceSynchronize()" 
                  << ", host copy_ops =" <<copy_ops.load() << std::endl;
    }

    std::cout << "Stopping, cleaning up...\n";

    for (auto &pd : devices) {
        HIP_CHECK(hipSetDevice(pd.devId));
        for (size_t s = 0; s < pd.streams.size(); ++s) {
            HIP_CHECK(hipStreamDestroy(pd.streams[s]));
            HIP_CHECK(hipFree(pd.d_inputs[s]));
            HIP_CHECK(hipFree(pd.d_outputs[s]));
            HIP_CHECK(hipHostFree(pd.h_pinned_src[s]));
            HIP_CHECK(hipHostFree(pd.h_pinned_dst[s]));
        }
    }

    g_stop.store(true);
    for (auto &t : workers) if (t.joinable()) t.join();

    std::cout << "All done.\n";
    return 0;
}
