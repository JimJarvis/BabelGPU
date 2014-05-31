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
// Ftype: float or double
// for example, functor: exp(x) 
// functor_1: exp(x + b) 
// functor_2: exp(a * x)
// functor_3: exp(a*x + b)
// default a = 1 and b = 0
#define GEN_linear_functor(name, Ftype) \
struct functor_##name##_##Ftype{ \
	__host__ __device__ Ftype operator()(const Ftype& x) const { return name(x); } \
}; \
struct functor_##name##_##Ftype##_1{ \
	const Ftype b; \
	functor_##name##_##Ftype##_1(Ftype _b = 0) : b(_b) {} \
	__host__ __device__ Ftype operator()(const Ftype& x) const { return name(x + b); } \
}; \
struct functor_##name##_##Ftype##_2{ \
	const Ftype a; \
	functor_##name##_##Ftype##_2(Ftype _a = 1) : a(_a) {} \
	__host__ __device__ Ftype operator()(const Ftype& x) const { return name(a * x); } \
}; \
struct functor_##name##_##Ftype##_3{ \
	const Ftype a, b; \
	functor_##name##_##Ftype##_3(Ftype _a = 1, Ftype _b = 0) : a(_a), b(_b) {} \
	__host__ __device__ Ftype operator()(const Ftype& x) const { return name(a * x + b); } \
};

// Macro defines corresponding thrust::transform for various linear unary functors
// gpu_exp_float(begin, size) is in place transformation, while gpu_exp_float(begin, size, out) writes to an output pointer.
#define GEN_transf_ftype(name, Ftype) \
GEN_linear_functor(name, Ftype); \
inline void gpu_##name##_##Ftype(device_ptr<Ftype> begin, int size, Ftype a = 1, Ftype b = 0) \
{ \
	if (a == 1 && b == 0) \
		transform(begin, begin + size, begin, functor_##name##_##Ftype()); \
	else if (a == 1) \
		transform(begin, begin + size, begin, functor_##name##_##Ftype##_1(b)); \
	else if (b == 0) \
		transform(begin, begin + size, begin, functor_##name##_##Ftype##_2(a)); \
	else \
		transform(begin, begin + size, begin, functor_##name##_##Ftype##_3(a, b)); \
} \
inline void gpu_##name##_##Ftype(device_ptr<Ftype> begin, int size, \
	device_ptr<Ftype> out, Ftype a = 1, Ftype b = 0) \
{ \
if (a == 1 && b == 0) \
	transform(begin, begin + size, out, functor_##name##_##Ftype()); \
	else if (a == 1) \
	transform(begin, begin + size, out, functor_##name##_##Ftype##_1(b)); \
	else if (b == 0) \
	transform(begin, begin + size, out, functor_##name##_##Ftype##_2(a)); \
	else \
	transform(begin, begin + size, out, functor_##name##_##Ftype##_3(a, b)); \
}


// Macro defines functors for linear transformations inside a binary function
// Ftype: float or double
// for example, functor: pow(x, p) 
// functor_1: pow(x + b, p) 
// functor_2: pow(a * x, p)
// functor_3: pow(a*x + b, p)
// default a = 1 and b = 0
#define GEN_linear_functor_2(name, Ftype) \
struct functor_##name##_##Ftype{ \
	const Ftype p; \
	functor_##name##_##Ftype(Ftype _p) : p(_p) {} \
	__host__ __device__ Ftype operator()(const Ftype& x) const { return name(x, p); } \
}; \
struct functor_##name##_##Ftype##_1{ \
	const Ftype p, b; \
	functor_##name##_##Ftype##_1(Ftype _p, Ftype _b = 0) : p(_p), b(_b) {} \
	__host__ __device__ Ftype operator()(const Ftype& x) const { return name(x + b, p); } \
}; \
struct functor_##name##_##Ftype##_2{ \
	const Ftype p, a; \
	functor_##name##_##Ftype##_2(Ftype _p, Ftype _a = 1) : p(_p), a(_a) {} \
	__host__ __device__ Ftype operator()(const Ftype& x) const { return name(a * x, p); } \
}; \
struct functor_##name##_##Ftype##_3{ \
	const Ftype p, a, b; \
	functor_##name##_##Ftype##_3(Ftype _p, Ftype _a = 1, Ftype _b = 0) : p(_p), a(_a), b(_b) {} \
	__host__ __device__ Ftype operator()(const Ftype& x) const { return name(a * x + b, p); } \
};

// Macro defines corresponding thrust::transform for various linear binary functors
// gpu_pow_float(begin, size) is in place transformation, while gpu_pow_float(begin, size, out) writes to an output pointer.
#define GEN_transf_ftype_2(name, Ftype) \
GEN_linear_functor_2(name, Ftype); \
inline void gpu_##name##_##Ftype(device_ptr<Ftype> begin, int size, Ftype p, Ftype a = 1, Ftype b = 0) \
{ \
	if (a == 1 && b == 0) \
		transform(begin, begin + size, begin, functor_##name##_##Ftype(p)); \
	else if (a == 1) \
		transform(begin, begin + size, begin, functor_##name##_##Ftype##_1(p, b)); \
	else if (b == 0) \
		transform(begin, begin + size, begin, functor_##name##_##Ftype##_2(p, a)); \
	else \
		transform(begin, begin + size, begin, functor_##name##_##Ftype##_3(p, a, b)); \
} \
inline void gpu_##name##_##Ftype(device_ptr<Ftype> begin, int size, \
	device_ptr<Ftype> out, Ftype p, Ftype a = 1, Ftype b = 0) \
{ \
if (a == 1 && b == 0) \
	transform(begin, begin + size, out, functor_##name##_##Ftype(p)); \
	else if (a == 1) \
	transform(begin, begin + size, out, functor_##name##_##Ftype##_1(p, b)); \
	else if (b == 0) \
	transform(begin, begin + size, out, functor_##name##_##Ftype##_2(p, a)); \
	else \
	transform(begin, begin + size, out, functor_##name##_##Ftype##_3(p, a, b)); \
}

// Combines float and double functions
#define GEN_transf(name) \
	GEN_transf_ftype(name, float); \
	GEN_transf_ftype(name, double);

#define GEN_transf_2(name) \
	GEN_transf_ftype_2(name, float); \
	GEN_transf_ftype_2(name, double);

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

//#ifndef _WIN32
//	GEN_transf(exp2);
//	GEN_transf(expm1); // exp - 1
//	GEN_transf(log1p); // ln + 1
//	GEN_transf(log2);
//	GEN_transf(cbrt); // cubic root
//	GEN_transf(hypot); // hypotenus
//	GEN_transf(erf); // error function
//	GEN_transf(erfc); // complementary error function
//	GEN_transf(tgamma); // gamma function
//	GEN_transf(lgamma); // log-gamma function
//	GEN_transf(acosh);
//	GEN_transf(asinh);
//	GEN_transf(atanh);
//#endif

	// gpu_min|max_float|double
#define GEN_minmax_ftype(name, Ftype) \
	inline Ftype gpu_##name##_##Ftype(device_ptr<Ftype> begin, int size) \
	{ \
		device_ptr<Ftype> m = name##_element(begin, begin + size); \
		return *m; \
	}
    #define GEN_minmax(name) \
	GEN_minmax_ftype(name, float); \
	GEN_minmax_ftype(name, double);

	GEN_minmax(min);
	GEN_minmax(max);

#define GEN_sum(Ftype) \
	inline Ftype gpu_sum_##Ftype(device_ptr<Ftype> begin, int size) \
	{ \
		return reduce(begin, begin+size, 0.0, thrust::plus<Ftype>()); \
	}

	GEN_sum(float); GEN_sum(double);

#define GEN_product(Ftype) \
	inline Ftype gpu_product_##Ftype(device_ptr<Ftype> begin, int size) \
	{ \
		return reduce(begin, begin+size, 1.0, thrust::multiplies<Ftype>()); \
	}

	GEN_product(float); GEN_product(double);

// dir = ascending: 1, descending -1
#define GEN_sort(Ftype) \
	inline void gpu_sort_##Ftype(device_ptr<Ftype> begin, int size, int dir = 1) \
	{ \
		if (dir > 0) \
			thrust::sort(begin, begin+size); \
		else /* descending sort */ \
			thrust::sort(begin, begin+size, greater<Ftype>()); \
	}

	GEN_sort(float); GEN_sort(double);


	// Fill the array with the same value
	inline void gpu_fill_float(device_ptr<float> begin, int size, float val)
	{ thrust::fill_n(begin, size, val); }
	inline void gpu_fill_double(device_ptr<double> begin, int size, double val)
	{ thrust::fill_n(begin, size, val); }

	inline void gpu_copy_float(device_ptr<float> begin, int size, device_ptr<float> out)
	{ thrust::copy(begin, begin + size, out); }
	inline void gpu_copy_double(device_ptr<double> begin, int size, device_ptr<double> out)
	{ thrust::copy(begin, begin + size, out); }

	// Swap two arrays
	inline void gpu_swap_float(device_ptr<float> begin, int size, device_ptr<float> out)
	{ thrust::swap_ranges(begin, begin + size, out); }
	inline void gpu_swap_double(device_ptr<double> begin, int size, device_ptr<double> out)
	{ thrust::swap_ranges(begin, begin + size, out); }

	// Utility function for java
	inline device_ptr<float> offset_float(device_ptr<float> begin, int offset)
	{ return begin + offset; }
	inline device_ptr<double> offset_double(device_ptr<double> begin, int offset)
	{ return begin + offset; }

	// Utility for setting blockDim and gridDim (1D). A block cannot have more than 1024 threads
	// number of threads needed, 2 output params
	inline void setKernelDim1D(int threads, dim3& gridDim, dim3& blockDim)
	{
		if (threads > 1024) // we don't have enough threads on a single block
		{
			gridDim.x = threads / 1024 + 1;
			blockDim.x = 1024;
		}
		else
			blockDim.x = threads;
	}

// Should be used inside kernel functions only
#define ThreadIndex1D(idx, limit) \
	int idx = blockIdx.x * blockDim.x + threadIdx.x; \
	if (idx >= limit) return; // out of bound


	// The specified col will be set to a specific value
	// negative 'colIdx' means counting from the last col (-n => col - n)
	inline void gpu_fill_col_float(device_ptr<float> begin, int row, int col, int colIdx, float val)
	{
		if (colIdx < 0)  colIdx += col;
		thrust::fill_n(begin + row * colIdx, row, val);
	}

	// Change a specific row of a column major matrix to a specific value
	// negative 'rowIdx' means counting from the last row (-n => row - n)
	__global__
	void gpu_fill_row_float_kernel(float *begin, int row, int col, int rowIdx, float val)
	{
			ThreadIndex1D(idx, col);

			begin += row * idx + rowIdx; // end of a column
			*begin = val; // set the value
		}

	// The specified row will be set to a specific value
	// negative 'rowIdx' means counting from the last row (-n => row - n)
	inline void gpu_fill_row_float(device_ptr<float> begin, int row, int col, int rowIdx, float val)
	{
		dim3 gridDim, blockDim;
		setKernelDim1D(col, gridDim, blockDim);

		if (rowIdx < 0) rowIdx += row;

		gpu_fill_row_float_kernel<<<gridDim, blockDim>>>(
			thrust::raw_pointer_cast(begin), row, col, rowIdx, val);
	}
}
#endif // try_h__