/*
* Babel project specific GPU instructions
*/
#ifndef babel_gpu_h__
#define babel_gpu_h__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include "my_gpu.h"
using namespace thrust;

namespace MyGpu
{
// I [y==j] - softmax_float(alpha_vec)
#define GEN_babel_id_minus_softmax(Ftype) \
inline void babel_id_minus_softmax_##Ftype(device_ptr<Ftype> begin, int size, int id) \
{ \
	Ftype mx = gpu_max_##Ftype(begin, size); \
	gpu_exp_##Ftype(begin, size, 1, -mx); \
	Ftype s = gpu_sum_##Ftype(begin, size); \
	gpu__##Ftype(begin, size, -1.0 / s, 0); \
	++ *(begin + id);  /* when at id, x = 1 - x */ \
}
GEN_babel_id_minus_softmax(float);
GEN_babel_id_minus_softmax(double);


// Second way to implement (id - softmax_float()), exactly the same numerical result. 
struct functor_id_minus_softmax_float_2
{
	const float b;
	functor_id_minus_softmax_float_2(float _b = 0) : b(_b) {}
	__host__ __device__ float operator()(const float& x) const { return -exp(x - b); }
};

// I [y==j] - softmax_float(alpha_vec)
// A = exp(A - (mx + log(sum(exp(A - mx))))
inline void babel_id_minus_softmax_float_2(device_ptr<float> begin, int size, int id)
{
	float mx = gpu_max_float(begin, size);

	float logsum = log(
		thrust::transform_reduce(begin, begin + size,
		functor_exp_float_1(-mx), 0.0f, thrust::plus<float>()));

	thrust::transform(begin, begin + size, begin, functor_id_minus_softmax_float_2(mx + logsum));
	++ *(begin + id);  // when at id, x = 1 - x
}

// mini-batch I [y==j] - softmax_float(alpha_vec)
__forceinline__ __global__ 
void babel_batch_kernel(float *begin, int row, int col, int *labels)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= col) return; // out of bound

	begin += idx * row; // beginning of a column

	// find max
	float mx = -1e20;
	for (int i = 0; i < row; i++)
		if (begin[i] > mx)
			mx = begin[i];
	// subtract max from each and do exp
	// also compute sum of these exp
	float sum = 0;
	for (int i = 0; i < row; i++)
	{
		begin[i] = exp(begin[i] - mx);
		sum += begin[i];
	}
	// -exp(..)/sum
	for (int i = 0; i < row; i++)
		begin[i] /= -sum;

	// Add 1 to the identity function
	++begin[labels[idx]];
}

inline void babel_batch_id_minus_softmax_float(
	device_ptr<float> begin, int row, int col, int *labels)
{
	dim3 gridDim, blockDim;
	if (col > 1024) // we don't have enough threads on a single block
	{
		gridDim.x = col / 1024 + 1;
		blockDim.x = 1024;
	}
	else
		blockDim.x = col;

	babel_batch_kernel<<<gridDim, blockDim>>>(
		thrust::raw_pointer_cast(begin), row, col, labels);
}

// for babel_batch_id_minus_softmax_float: deal with JavaCpp int pointer
inline int *copy_host_to_device(int *host, int size)
{
	int *device;
	size *= sizeof(int);
	cudaMalloc((void **)&device, size);
	cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
	return device;
}

inline void gpu_free(int *device)
{
	cudaFree(device);
}

}

#endif // babel_gpu_h__