/*
* Port the Thrust library to Java with JavaCpp tool. 
*/

#ifndef my_thrust__
#define my_thrust__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES // otherwise cmath doesn't have M_PI
#include <cmath>
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
using namespace thrust;

// Macro defines functors for linear transformations inside an elementary unary function
// Ftype: float or double
// for example, functor: exp(x) 
// functor_1: exp(x + b) 
// functor_2: exp(a * x)
// functor_3: exp(a*x + b)
// default a = 1 and b = 0
#define GEN_linear_functor(name) \
	template <typename T> \
struct functor_##name{ \
	__host__ __device__ T operator()(const T& x) const { return name(x); } \
}; \
	template <typename T> \
struct functor_##name##_1{ \
	const T b; \
	functor_##name##_1(T _b = 0) : b(_b) {} \
	__host__ __device__ T operator()(const T& x) const { return name(x + b); } \
}; \
	template <typename T> \
struct functor_##name##_2{ \
	const T a; \
	functor_##name##_2(T _a = 1) : a(_a) {} \
	__host__ __device__ T operator()(const T& x) const { return name(a * x); } \
}; \
	template <typename T> \
struct functor_##name##_3{ \
	const T a, b; \
	functor_##name##_3(T _a = 1, T _b = 0) : a(_a), b(_b) {} \
	__host__ __device__ T operator()(const T& x) const { return name(a * x + b); } \
};

// Macro defines corresponding thrust::transform for various linear unary functors
// gpu_exp_float(begin, size) is in place transformation, while gpu_exp_float(begin, size, out) writes to an output pointer.
#define GEN_transf(name) \
GEN_linear_functor(name); \
template <typename T> \
inline void gpu_##name(device_ptr<T> begin, int size, \
								device_ptr<T> out, T a = 1, T b = 0) \
{ \
if (a == 1 && b == 0) \
	transform(begin, begin + size, out, functor_##name<T>()); \
	else if (a == 1) \
	transform(begin, begin + size, out, functor_##name##_1<T>(b)); \
	else if (b == 0) \
	transform(begin, begin + size, out, functor_##name##_2<T>(a)); \
	else \
	transform(begin, begin + size, out, functor_##name##_3<T>(a, b)); \
} \
/* Overload in == out */ \
template <typename T> \
inline void gpu_##name(device_ptr<T> begin, int size, T a = 1, T b = 0) \
{ \
	gpu_##name<T>(begin, size, begin, a, b); \
}


// Macro defines functors for linear transformations inside a binary function
// T: float or double
// for example, functor: pow(x, p) 
// functor_1: pow(x + b, p) 
// functor_2: pow(a * x, p)
// functor_3: pow(a*x + b, p)
// default a = 1 and b = 0
#define GEN_linear_functor_2(name) \
template <typename T> \
struct functor_##name{ \
	const T p; \
	functor_##name(T _p) : p(_p) {} \
	__host__ __device__ T operator()(const T& x) const { return name(x, p); } \
}; \
template <typename T> \
struct functor_##name##_1{ \
	const T p, b; \
	functor_##name##_1(T _p, T _b = 0) : p(_p), b(_b) {} \
	__host__ __device__ T operator()(const T& x) const { return name(x + b, p); } \
}; \
template <typename T> \
struct functor_##name##_2{ \
	const T p, a; \
	functor_##name##_2(T _p, T _a = 1) : p(_p), a(_a) {} \
	__host__ __device__ T operator()(const T& x) const { return name(a * x, p); } \
}; \
template <typename T> \
struct functor_##name##_3{ \
	const T p, a, b; \
	functor_##name##_3(T _p, T _a = 1, T _b = 0) : p(_p), a(_a), b(_b) {} \
	__host__ __device__ T operator()(const T& x) const { return name(a * x + b, p); } \
};

// Macro defines corresponding thrust::transform for various linear binary functors
// gpu_pow_float(begin, size) is in place transformation, while gpu_pow_float(begin, size, out) writes to an output pointer.
#define GEN_transf_2(name) \
GEN_linear_functor_2(name); \
template <typename T> \
inline void gpu_##name(device_ptr<T> begin, int size, \
								device_ptr<T> out, T p, T a = 1, T b = 0) \
{ \
if (a == 1 && b == 0) \
	transform(begin, begin + size, out, functor_##name<T>(p)); \
	else if (a == 1) \
	transform(begin, begin + size, out, functor_##name##_1<T>(p, b)); \
	else if (b == 0) \
	transform(begin, begin + size, out, functor_##name##_2<T>(p, a)); \
	else \
	transform(begin, begin + size, out, functor_##name##_3<T>(p, a, b)); \
} \
/* Overload in == out */	\
template <typename T> \
inline void gpu_##name(device_ptr<T> begin, int size, T p, T a = 1, T b = 0) \
{ \
	gpu_##name<T>(begin, size, begin, p, a, b); \
}

namespace MyGpu
{
	// Generate unary transform functions
	// Exp and logs
	GEN_transf(exp);
	GEN_transf(log);
	GEN_transf(log10);
	GEN_transf(sqrt);

	// Trigs
	GEN_transf(cos);
	GEN_transf(sin);
	GEN_transf(tan);
	GEN_transf(acos);
	GEN_transf(asin);
	GEN_transf(atan);
	GEN_transf(cosh);
	GEN_transf(sinh);
	GEN_transf(tanh);

	// Other
	GEN_transf(fabs); // abs()
	GEN_transf(floor);
	GEN_transf(ceil);
	GEN_transf(); // gpu__float(), for plain linear transformation

	// Generate binary transform functions
	GEN_transf_2(pow);
	GEN_transf_2(fmod);

	/* Other non-standard functions */
	template<typename T> __host__ __device__
	inline T sigmoid(T x) { return 1.0 / (1 + exp(-x)); }
	GEN_transf(sigmoid);
	// Sigmoid derivative: x .* (1 - x)
	template<typename T> __host__ __device__
	inline T sigmoid_deriv(T x) { return x * (1 - x); }
	GEN_transf(sigmoid_deriv);

	// simple square routine
	template<typename T> __host__ __device__
	inline T square(T x) { return x * x; }
	GEN_transf(square);
	// simple cube routine
	template<typename T> __host__ __device__
	inline T cube(T x) { return x * x * x; }
	GEN_transf(cube);
	// simple reciprocal routine
	template<typename T> __host__ __device__
	inline T reciprocal(T x) { return 1.0/x; }
	GEN_transf(reciprocal);

	// Random distribution generators
	// Cauchy CDF: given a uniform random var, transform it to be cauchy
	template<typename T> __host__ __device__
	inline T cauchy(T x) { return tan(M_PI * (x - 0.5)); }
	GEN_transf(cauchy);

	// Laplacian CDF: given a uniform random var, transform it to be lap
	// if < 0, return -1; > 0, return 1
	template<typename T> __host__ __device__
	inline T signum(T val)
	{
		return (0 < val) - (val < 0);
	}
	template<typename T> __host__ __device__
	inline T laplacian(T x)
	{ 
		x -= 0.5;
		return -signum(x) * log(1 - 2 * fabs(x));
	}
	GEN_transf(laplacian);
	GEN_transf(signum);


	///////***** OTHER functions *****///////
	template <typename T>
	inline T gpu_min(device_ptr<T> begin, int size) 
	{
		device_ptr<T> m = min_element(begin, begin + size);
		return *m;
	}

	template <typename T>
	inline T gpu_max(device_ptr<T> begin, int size) 
	{
		device_ptr<T> m = max_element(begin, begin + size);
		return *m;
	}

// dir = ascending: 1, descending -1
	template <typename T>
	inline void gpu_sort(device_ptr<T> begin, int size, int dir = 1)
	{
		if (dir > 0)
			thrust::sort(begin, begin+size);
		else /* descending sort */
			thrust::sort(begin, begin+size, greater<T>());
	}

	template <typename T>
	struct functor_scale_mult { 
		const T s; 
		functor_scale_mult(T _s = 1) : s(_s) {} 
		__host__ __device__  
		T operator()(const T& x, const T& y) const { return s * x * y; } 
	}; 
	/* begin2 = begin1 .* begin2 */ 
	template <typename T>
	inline void gpu_dot_mult(device_ptr<T> begin1, int size, device_ptr<T> begin2, float scalor = 1) 
	{
		transform(begin1, begin1 + size, begin2, begin2, functor_scale_mult<T>(scalor));
	}
	/* out = begin1 .* begin2 */ 
	template <typename T>
	inline void gpu_dot_mult(device_ptr<T> begin1, int size, device_ptr<T> begin2, device_ptr<T> out, float scalor = 1) 
	{
		transform(begin1, begin1 + size, begin2, out, functor_scale_mult<T>(scalor));
	}

	template <typename T>
	inline T gpu_sum(device_ptr<T> begin, int size)
	{ return reduce(begin, begin+size, 0.0, thrust::plus<T>()); }

	template <typename T>
	inline T gpu_product(device_ptr<T> begin, int size)
	{ return reduce(begin, begin+size, 1.0, thrust::multiplies<T>()); }

	// Fill the array with the same value
	template <typename T>
	inline void gpu_fill(device_ptr<T> begin, int size, T val)
	{ thrust::fill_n(begin, size, val); }

	template <typename T>
	inline void gpu_copy(device_ptr<T> begin, int size, device_ptr<T> out)
	{ thrust::copy(begin, begin + size, out); }

	// Swap two arrays
	template <typename T>
	inline void gpu_swap(device_ptr<T> begin, int size, device_ptr<T> out)
	{ thrust::swap_ranges(begin, begin + size, out); }

	// Utility function for java
	template <typename T>
	inline device_ptr<T> offset(device_ptr<T> begin, int offset)
	{ return begin + offset; }

	// Set or update a single value on device
	// for gradCheck perturb()
	template <typename T>
	inline void gpu_set_single(device_ptr<T> begin, int offset, T newVal) 
	{ begin[offset] = newVal; } 

	template <typename T>
	inline void gpu_incr_single(device_ptr<T> begin, int offset, T incrVal) \
	{ begin[offset] += incrVal; }

	///////***** Deal with int, float, and double raw GPU pointers  *****///////
	template <typename T>
	inline T* offset(T *begin, int offset) { return begin + offset; } 
	template <typename T>
	inline void free_device(T *device) { cudaFree(device); } 
	template <typename T>
	inline void free_host(T *host) { free(host); } 

	template <typename T>
	inline T *copy_host_to_device(T *host, int size) 
	{ 
		T *device; size *= sizeof(T); 
		cudaMalloc((void **)&device, size); 
		cudaMemcpy(device, host, size, cudaMemcpyHostToDevice); 
		return device; 
	} 
	template <typename T>
	inline T *copy_device_to_host(T *device, int size) 
	{ 
		size *= sizeof(T); 
		T *host = (T *)malloc(size); 
		cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost); 
		return host; 
	} 
	/* Can be used to copy directly to a primitive array (JavaCpp @Ptr) */ 
	template <typename T>
	inline void copy_device_to_host(T *device, T host[], int offset, int size) 
	{ 
		cudaMemcpy(host + offset, device, size * sizeof(T), cudaMemcpyDeviceToHost); 
	}

#define GEN_malloc(Ftype) \
	inline Ftype* malloc_device_##Ftype(int size, bool memsetTo0) \
	{  \
		Ftype *device; \
		cudaMalloc((void **)&device, size * sizeof(Ftype)); \
		if (memsetTo0) \
			cudaMemset(device, 0, size * sizeof(Ftype)); \
		return device; \
	} 
	GEN_malloc(int);
	GEN_malloc(float);
	GEN_malloc(double);


	/**********************************************/
	/* More sophisticated functions for machine learning  */
	/**********************************************/
	/* softmax(alpha_vec) */ 
	template <typename T>
	inline void gpu_softmax(device_ptr<T> begin, int size, device_ptr<T> out) 
	{ 
		T mx = gpu_max<T>(begin, size);
		gpu_exp<T>(begin, size, out, 1, -mx); 
		T s = gpu_sum<T>(out, size);
		gpu_<T>(out, size, 1.0 / s, 0);
	} 
	/*Overload: in == out*/ 
	template <typename T>
	inline void gpu_softmax(device_ptr<T> begin, int size) 
	{
		gpu_softmax<T>(begin, size, begin);
	}

	/* softmax(alpha_vec) - I [y==j] */ 
	template <typename T>
	inline void gpu_softmax_minus_id(device_ptr<T> begin, int size, device_ptr<T> out, int label) 
	{ 
		gpu_softmax<T>(begin, size, out);
		-- *(out + label);  /* when at id, x -= 1 */ 
	} 
	/*Overload: in == out*/ 
	template <typename T>
	inline void gpu_softmax_minus_id(device_ptr<T> begin, int size, int label) 
	{
		gpu_softmax_minus_id<T>(begin, size, begin, label);
	}

	/* softmax(alpha_vec) at only the correct label. 'outProb' is a 1 float device_ptr. 'begin' data won't be changed */ 
	template <typename T>
	inline void gpu_softmax_at_label(device_ptr<T> begin, int size, int label, device_ptr<T> outProb) 
	{ 
		T mx = gpu_max<T>(begin, size); 
		T expSum = thrust::transform_reduce(begin, begin + size, 
											functor_exp_1<T>(-mx), 0.0, thrust::plus<T>());
		outProb[0] = exp(begin[label] - mx) / expSum; 
	}

	// Sum of log of correct label probability
	// input: a float array computed by softmax()
	template <typename T>
	inline T gpu_log_sum(device_ptr<T> begin, int size)
	{
		return thrust::transform_reduce(begin, begin + size,
										functor_log<T>(), 0.0, thrust::plus<T>());
	}

	// softmax(alpha_vec) - I [y==j]
	// A = exp(A - (mx + log(sum(exp(A - mx))))
	// Deprecated
	inline void gpu_softmax_minus_id_2(device_ptr<float> begin, int size, int id)
	{
		float mx = gpu_max<float>(begin, size);

		float logsum = log(
			thrust::transform_reduce(begin, begin + size,
			functor_exp_1<float>(-mx), 0.0, thrust::plus<float>()));

		gpu_exp<float>(begin, size, 1, -(mx + logsum));
		-- *(begin + id);  // when at id, x -= 1
	}
}
#endif // my_thrust__