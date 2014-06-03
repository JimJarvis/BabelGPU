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
#define GEN_babel_softmax(Ftype) \
	/* I [y==j] - softmax(alpha_vec) */ \
inline void babel_id_minus_softmax(device_ptr<Ftype> begin, int size, int label) \
{ \
	Ftype mx = gpu_max_##Ftype(begin, size); \
	gpu_exp_##Ftype(begin, size, 1, -mx); \
	Ftype s = gpu_sum_##Ftype(begin, size); \
	gpu__##Ftype(begin, size, -1.0 / s, 0); \
	++ *(begin + label);  /* when at id, x = 1 - x */ \
} \
	/* softmax(alpha_vec) */ \
inline void babel_softmax(device_ptr<Ftype> begin, int size) \
{ \
	Ftype mx = gpu_max_##Ftype(begin, size); \
	gpu_exp_##Ftype(begin, size, 1, -mx); \
	Ftype s = gpu_sum_##Ftype(begin, size); \
	gpu__##Ftype(begin, size, 1.0 / s, 0); \
} \
	/* softmax(alpha_vec) at only the correct label. 'out' is a 1 float device_ptr */ \
inline void babel_softmax(device_ptr<Ftype> begin, int size, int label, device_ptr<Ftype> out) \
{ \
	Ftype mx = gpu_max_##Ftype(begin, size); \
	Ftype expSum = thrust::transform_reduce(begin, begin + size, \
		functor_exp_##Ftype##_1(-mx), 0.0, thrust::plus<Ftype>()); \
	out[0] = exp(begin[label] - mx) / expSum; \
}

GEN_babel_softmax(float);
GEN_babel_softmax(double);


///// Second way to implement (id - softmax()), exactly the same numerical result. 
struct functor_id_minus_softmax_2
{
	const float b;
	functor_id_minus_softmax_2(float _b = 0) : b(_b) {}
	__host__ __device__ float operator()(const float& x) const { return -exp(x - b); }
};

// I [y==j] - softmax(alpha_vec)
// A = exp(A - (mx + log(sum(exp(A - mx))))
inline void babel_id_minus_softmax_2(device_ptr<float> begin, int size, int id)
{
	float mx = gpu_max_float(begin, size);

	float logsum = log(
		thrust::transform_reduce(begin, begin + size,
		functor_exp_float_1(-mx), 0.0, thrust::plus<float>()));

	thrust::transform(begin, begin + size, begin, functor_id_minus_softmax_2(mx + logsum));
	++ *(begin + id);  // when at id, x = 1 - x
}

///// mini-batch I [y==j] - softmax(alpha_vec)
__global__ 
void babel_batch_id_minus_softmax_kernel(
		float *begin, int row, int col, int *labels)
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

inline void babel_batch_id_minus_softmax(
	device_ptr<float> begin, int row, int col, int *labels)
{
	if (col == 1) // use the thrust version
	{
		int label;
		cudaMemcpy(&label, labels, sizeof(int), cudaMemcpyDeviceToHost);
		babel_id_minus_softmax(begin, row, label);
	}
	else // real batch
	{
		dim3 gridDim, blockDim;
		setKernelDim1D(col, gridDim, blockDim);

		babel_batch_id_minus_softmax_kernel<<<gridDim, blockDim>>>(
			thrust::raw_pointer_cast(begin), row, col, labels);
	}
}

///// mini-batch softmax(alpha_vec)
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

// Computes the mini-batch softmax probability distribution for the full matrix
inline void babel_batch_softmax(
	device_ptr<float> begin, int row, int col)
{
	if (col == 1) // use the thrust version
		babel_softmax(begin, row);
	else // real batch
	{
		dim3 gridDim, blockDim;
		setKernelDim1D(col, gridDim, blockDim);

		babel_batch_softmax_kernel<<<gridDim, blockDim >>>(
			thrust::raw_pointer_cast(begin), row, col);
	}
}

///// mini-batch softmax(alpha_vec) that writes to 'out' only the probability at the correct label
__global__
void babel_batch_softmax_kernel(
	float *begin, int row, int col, float *out, int *labels)
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
	float numerator, tmp;
	int label = labels[idx];
	for (int i = 0; i < row; i++)
	{
		tmp = exp(begin[i] - mx);
		if (i == label) numerator = tmp;
		sum += tmp;
	}
	out[idx] = numerator / sum;
}

// Computes the mini-batch softmax probability distribution for the full matrix
// writes to 'out' only the probability at the correct label
// int *labels is on GPU
inline void babel_batch_softmax(
	device_ptr<float> begin, int row, int col, device_ptr<float> out, int *labels)
{
	if (col == 1) // use the thrust version
	{
		int label;
		cudaMemcpy(&label, labels, sizeof(int), cudaMemcpyDeviceToHost);
		babel_softmax(begin, row, label, out);
	}
	else // real batch
	{
		dim3 gridDim, blockDim;
		setKernelDim1D(col, gridDim, blockDim);

		babel_batch_softmax_kernel << <gridDim, blockDim >> >(
			thrust::raw_pointer_cast(begin), row, col, thrust::raw_pointer_cast(out), labels);
	}
}

///// Fill 'outLabels' with the label corresponding to the maximum probability of a column
__global__
void babel_best_label_kernel(
	float *begin, int row, int col, int *outLabels)
{
	ThreadIndex1D(idx, col);

	begin += idx * row; // beginning of a column

	// find max
	float mx = -1e20;
	int maxLabel = 0;
	for (int i = 0; i < row; i++)
		if (begin[i] > mx)
		{
			mx = begin[i];
			maxLabel = i;
		}
	outLabels[idx] = maxLabel;
}

// Fill 'outLabels' with the label corresponding to the maximum probability of a column
// outLabels is filled out on GPU
inline void babel_best_label(
	device_ptr<float> begin, int row, int col, int *outLabels)
{
	dim3 gridDim, blockDim;
	setKernelDim1D(col, gridDim, blockDim);

	babel_best_label_kernel<<<gridDim, blockDim>>>(
		thrust::raw_pointer_cast(begin), row, col, outLabels);
}

// Sum of log of correct label probability
// input: a float array computed by softmax()
inline float babel_log_prob(device_ptr<float> begin, int size)
{
	return thrust::transform_reduce(begin, begin + size,
						 functor_log_float(), 0.0, thrust::plus<float>());
}

}

#endif // babel_gpu_h__