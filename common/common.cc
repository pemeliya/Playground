
#include "common/common_utils.hpp"

XLogMessage::XLogMessage(const char* fname, int line, int/* severity*/) :
  fname_(fname), line_(line) { }

XLogMessage::~XLogMessage() {
  fprintf(stderr, "[%s:%d] %s\n", fname_, line_, str().c_str());
}

GpuTimer::GpuTimer()
{
  (void)cudaEventCreate(&start);
  (void)cudaEventCreate(&stop);
}

GpuTimer::~GpuTimer()
{
  (void)cudaEventDestroy(start);
  (void)cudaEventDestroy(stop);
}

void GpuTimer::Start()
{
  (void)cudaEventRecord(start, 0);
}

void GpuTimer::Stop()
{
  (void)cudaEventRecord(stop, 0);
}

float GpuTimer::ElapsedMillis()
{
  float elapsed;
  (void)cudaEventSynchronize(stop);
  (void)cudaEventElapsedTime(&elapsed, start, stop);
  return elapsed;
}

GPUStream::GPUStream(int priority) 
{
  if (priority == 0) {
    CHK(cudaStreamCreateWithFlags(&handle_, cudaStreamDefault)) 
  } else {
    CHK(cudaStreamCreateWithPriority(&handle_, cudaStreamDefault, priority))
  }
}

GPUStream::~GPUStream() {
  (void)cudaStreamDestroy(handle_);
}

void DeviceInit(int dev)
{
    int deviceCount;
    CHK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        throw std::runtime_error("No devices supporting CUDA/HIP");
    }

    CHK(cudaSetDevice(dev));

    std::size_t device_free_physmem = 0, device_total_physmem = 0;
    CHK(cudaMemGetInfo(&device_free_physmem, &device_total_physmem)); // this fails on rocm-5.7.0

    cudaDeviceProp deviceProp;

    CHK(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 1) {
        throw std::runtime_error("Device does not support CUDA/HIP");
    }

    auto device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;
    {
        //printf("%llu --- %llu\n", device_free_physmem, device_total_physmem);
        printf("Using device %d: %s ( SM%d, %d SMs, "
                        "%lld free / %lld total MB physmem, "
                        "%.3f GB/s @ %d kHz mem clock, ECC %s)\n",
                    dev,
                    deviceProp.name,
                    deviceProp.major * 100 + deviceProp.minor * 10,
                    deviceProp.multiProcessorCount,
                    (unsigned long long) device_free_physmem / 1024 / 1024,
                    (unsigned long long) device_total_physmem / 1024 / 1024,
                    device_giga_bandwidth,
                    deviceProp.memoryClockRate,
                    (deviceProp.ECCEnabled) ? "on" : "off");
        fflush(stdout);
    }
}