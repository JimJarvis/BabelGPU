/*
* Babel project specific GPU instructions
*/
#ifndef babel_gpu_h__
#define babel_gpu_h__

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
}

#endif // babel_gpu_h__