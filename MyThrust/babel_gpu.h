#ifndef babel_gpu_h__
#define babel_gpu_h__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
using namespace thrust;

namespace MyGpu
{
// I [y==j] - softmax(alpha)
struct babel_id_minus_softmax_functor
{
	__host__ __device__ float operator()(const float& x) const
	{
		return log(x);
	}
};

// I [y==j] - softmax(alpha)
inline int babel_id_minus_softmax(device_ptr<float> begin, device_ptr<float> end, int id)
{
	//float mx = gpu_max_float(begin, end);
	return 66;

}
}

#endif // babel_gpu_h__