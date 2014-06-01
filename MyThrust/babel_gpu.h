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
// I [y==j] - softmax(alpha_vec)
// and simply softmax()
#define GEN_babel_softmax(Ftype) \
inline void babel_id_minus_softmax_##Ftype(device_ptr<Ftype> begin, int size, int id) \
{ \
	Ftype mx = gpu_max_##Ftype(begin, size); \
	gpu_exp_##Ftype(begin, size, 1, -mx); \
	Ftype s = gpu_sum_##Ftype(begin, size); \
	gpu__##Ftype(begin, size, -1.0 / s, 0); \
	++ *(begin + id);  /* when at id, x = 1 - x */ \
} \
inline void babel_softmax_##Ftype(device_ptr<Ftype> begin, int size) \
{ \
	Ftype mx = gpu_max_##Ftype(begin, size); \
	gpu_exp_##Ftype(begin, size, 1, -mx); \
	Ftype s = gpu_sum_##Ftype(begin, size); \
	gpu__##Ftype(begin, size, 1.0 / s, 0); \
}
GEN_babel_softmax(float);
GEN_babel_softmax(double);


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
__global__ 
void babel_batch_id_minus_softmax_kernel(float *begin, int row, int col, int *labels)
{
	ThreadIndex1D(idx, col);

	begin += idx * row; // beginning of a column

	// find max
	float mx = -1e20;
	for (int i = 0; i < row; i++)
		if (begin[i] > mx) mx = begin[i];
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
	if (col == 1) // use the thrust version
	{
		int *label = (int *) malloc(sizeof(int));
		cudaMemcpy(label, labels, sizeof(int), cudaMemcpyDeviceToHost);
		babel_id_minus_softmax_float(begin, row, *label);
		free(label);
	}
	else // real batch
	{
		dim3 gridDim, blockDim;
		setKernelDim1D(col, gridDim, blockDim);

		babel_batch_id_minus_softmax_kernel<<<gridDim, blockDim>>>(
			thrust::raw_pointer_cast(begin), row, col, labels);
	}
}

// mini-batch softmax_float(alpha_vec)
__global__
void babel_batch_softmax_kernel(float *begin, int row, int col)
{
	ThreadIndex1D(idx, col);

	begin += idx * row; // beginning of a column

	// find max
	float mx = -1e20;
	for (int i = 0; i < row; i++)
	if (begin[i] > mx) mx = begin[i];
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
		begin[i] /= sum;
}

// Computes the mini-batch softmax probability distribution
inline void babel_batch_softmax_float(
	device_ptr<float> begin, int row, int col)
{
	if (col == 1) // use the thrust version
	{
		babel_softmax_float(begin, row);
	}
	else // real batch
	{
		dim3 gridDim, blockDim;
		setKernelDim1D(col, gridDim, blockDim);

		babel_batch_softmax_kernel<<<gridDim, blockDim >>>(
			thrust::raw_pointer_cast(begin), row, col);
	}
}

}

#endif // babel_gpu_h__