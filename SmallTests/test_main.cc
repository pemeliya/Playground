
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <numeric>
#include <random>
#include "common/example_utils.hpp"


__global__ void powerf32_kernel(const float *X, const float *Y, float *out) {

  uint32_t thid = threadIdx.x;
  auto x = X[thid], y = Y[thid];
  out[thid] = powf(x, y);
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> AllSignedPairs(
    std::initializer_list< T > abs_vals) {
  std::vector<T> ys, xs;
  const size_t n = 4 * abs_vals.size() * abs_vals.size();
  ys.reserve(n);
  xs.reserve(n);
  for (auto abs_y : abs_vals) {
    for (auto y : {-abs_y, abs_y}) {
      for (auto abs_x : abs_vals) {
        for (auto x : {-abs_x, abs_x}) {
          ys.push_back(y);
          xs.push_back(x);
        }
      }
    }
  }
  return {xs, ys};
}

void runPowf32() 
{
  float xmax = std::numeric_limits< float >::max(),
        xeps = 1.0f + std::numeric_limits< float >::epsilon();
  
  auto [xs,ys] = AllSignedPairs({xeps, xmax});
  HVector< float > X(std::move(xs)), Y(std::move(ys)), Z(X.size());

  X.copyHToD();
  Y.copyHToD();

  uint32_t nblocks = 1, nthreads = X.size();
  powerf32_kernel<<<nblocks, nthreads>>>(X.devPtr, Y.devPtr, Z.devPtr);

  CHK(cudaPeekAtLastError());
  (void)cudaDeviceSynchronize();                       
  Z.copyDToH();

  VLOG(std::setprecision(8));
  for(int i = 0; i < Z.size(); i++) {
    auto z = std::pow(X[i], Y[i]);
    VLOG("pow(" << X[i] << ", " << Y[i] << ") = " << Z[i] << " truth: " << z);
  }
}

int main() try 
{
   runPowf32();
}
catch(std::exception& ex) {
  VLOG("Exception: " << ex.what());
}
