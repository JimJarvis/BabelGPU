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
// I [y==j] - softmax(alpha_vec)
inline void babel_id_minus_softmax(device_ptr<float> begin, int size, int id)
{
	float mx = gpu_max_float(begin, size);
	gpu_exp_float(begin, size, 1, -mx);
	float s = gpu_sum_float(begin, size);
	gpu__float(begin, size, -1.0f / s, 0);
	++ *(begin + id);  // when at id, x = 1 - x
}
}

#endif // babel_gpu_h__