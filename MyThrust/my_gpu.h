/*
* Port the Thrust library to Java with JavaCpp tool. 
*/

#ifndef try_h__
#define try_h__

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

// Macro defines functors for linear transformations inside an elementary unary function
// for example, functor: exp(x) 
// functor_1: exp(x + b) 
// functor_2: exp(a * x)
// functor_3: exp(a*x + b)
// default a = 1 and b = 0
#define GEN_LINEAR_FLOAT_FUNCTOR(name) \
struct functor_##name##_float{ \
	__host__ __device__ float operator()(const float& x) const { return name(x); } \
}; \
struct functor_##name##_float_1{ \
	const float b; \
	functor_##name##_float_1(float _b = 0) : b(_b) {} \
	__host__ __device__ float operator()(const float& x) const { return name(x + b); } \
}; \
struct functor_##name##_float_2{ \
	const float a; \
	functor_##name##_float_2(float _a = 1) : a(_a) {} \
	__host__ __device__ float operator()(const float& x) const { return name(a * x); } \
}; \
struct functor_##name##_float_3{ \
	const float a, b; \
	functor_##name##_float_3(float _a = 1, float _b = 0) : a(_a), b(_b) {} \
	__host__ __device__ float operator()(const float& x) const { return name(a * x + b); } \
};

// Macro defines corresponding thrust::transform for various linear unary functors
#define GEN_FLOAT_TRANS(name) \
GEN_LINEAR_FLOAT_FUNCTOR(name); \
inline void gpu_##name##_float(device_ptr<float> begin, device_ptr<float> end, float a = 1, float b = 0) \
{ \
	if (a == 1 && b == 0) \
		transform(begin, end, begin, functor_##name##_float()); \
	else if (a == 1) \
		transform(begin, end, begin, functor_##name##_float_1(b)); \
	else if (b == 0) \
		transform(begin, end, begin, functor_##name##_float_2(a)); \
	else \
		transform(begin, end, begin, functor_##name##_float_3(a, b)); \
}


// Macro defines functors for linear transformations inside a binary function
// for example, functor: pow(x, p) 
// functor_1: pow(x + b, p) 
// functor_2: pow(a * x, p)
// functor_3: pow(a*x + b, p)
// default a = 1 and b = 0
#define GEN_LINEAR_FLOAT_FUNCTOR_2(name) \
struct functor_##name##_float{ \
	const float p; \
	functor_##name##_float(float _p) : p(_p) {} \
	__host__ __device__ float operator()(const float& x) const { return name(x, p); } \
}; \
struct functor_##name##_float_1{ \
	const float p, b; \
	functor_##name##_float_1(float _p, float _b = 0) : p(_p), b(_b) {} \
	__host__ __device__ float operator()(const float& x) const { return name(x + b, p); } \
}; \
struct functor_##name##_float_2{ \
	const float p, a; \
	functor_##name##_float_2(float _p, float _a = 1) : p(_p), a(_a) {} \
	__host__ __device__ float operator()(const float& x) const { return name(a * x, p); } \
}; \
struct functor_##name##_float_3{ \
	const float p, a, b; \
	functor_##name##_float_3(float _p, float _a = 1, float _b = 0) : p(_p), a(_a), b(_b) {} \
	__host__ __device__ float operator()(const float& x) const { return name(a * x + b, p); } \
};

// Macro defines corresponding thrust::transform for various linear binary functors
#define GEN_FLOAT_TRANS_2(name) \
GEN_LINEAR_FLOAT_FUNCTOR_2(name); \
inline void gpu_##name##_float(device_ptr<float> begin, device_ptr<float> end, float p, float a = 1, float b = 0) \
{ \
	if (a == 1 && b == 0) \
		transform(begin, end, begin, functor_##name##_float(p)); \
	else if (a == 1) \
		transform(begin, end, begin, functor_##name##_float_1(p, b)); \
	else if (b == 0) \
		transform(begin, end, begin, functor_##name##_float_2(p, a)); \
	else \
		transform(begin, end, begin, functor_##name##_float_3(p, a, b)); \
}

namespace MyGpu
{
// Generate unary transform functions
// Exp and logs
GEN_FLOAT_TRANS(exp);
GEN_FLOAT_TRANS(log);
GEN_FLOAT_TRANS(log10);
GEN_FLOAT_TRANS(sqrt);

// Trigs
GEN_FLOAT_TRANS(cos);
GEN_FLOAT_TRANS(sin);
GEN_FLOAT_TRANS(tan);
GEN_FLOAT_TRANS(acos);
GEN_FLOAT_TRANS(asin);
GEN_FLOAT_TRANS(atan);
GEN_FLOAT_TRANS(cosh);
GEN_FLOAT_TRANS(sinh);
GEN_FLOAT_TRANS(tanh);

// Other
GEN_FLOAT_TRANS(fabs); // abs()
GEN_FLOAT_TRANS(floor);
GEN_FLOAT_TRANS(ceil);
GEN_FLOAT_TRANS(); // gpu__float(), for plain linear transformation

// Generate binary transform functions
GEN_FLOAT_TRANS_2(pow);
GEN_FLOAT_TRANS_2(fmod);

// MSVC doesn't yet fully support C++11, these only work on linux
#ifndef _WIN32
GEN_FLOAT_TRANS(exp2);
GEN_FLOAT_TRANS(expm1); // exp - 1
GEN_FLOAT_TRANS(log1p); // ln + 1
GEN_FLOAT_TRANS(log2);
GEN_FLOAT_TRANS(cbrt); // cubic root
GEN_FLOAT_TRANS(hypot); // hypotenus
GEN_FLOAT_TRANS(erf); // error function
GEN_FLOAT_TRANS(erfc); // complementary error function
GEN_FLOAT_TRANS(tgamma); // gamma function
GEN_FLOAT_TRANS(lgamma); // log-gamma function
GEN_FLOAT_TRANS(acosh);
GEN_FLOAT_TRANS(asinh);
GEN_FLOAT_TRANS(atanh);
#endif

inline float gpu_max_float(device_ptr<float> begin, device_ptr<float> end)
{
	device_ptr<float> m = max_element(begin, end);
	return *m;
}

inline float gpu_sum_float(device_ptr<float> begin, device_ptr<float> end)
{
	return reduce(begin, end, 0.0f, thrust::plus<float>());
}

inline float gpu_product_float(device_ptr<float> begin, device_ptr<float> end)
{
	return reduce(begin, end, 1.0f, thrust::multiplies<float>());
}

}

#endif // try_h__
