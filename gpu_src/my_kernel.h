#ifndef my_kernel_h__
#define my_kernel_h__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES // otherwise cmath doesn't have M_PI
#include "my_gpu.h"

namespace MyGpu
{
    // Should be used inside kernel functions only
#define ThreadIndex1D(idx, limit) \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= limit) return; // out of bound

    // because kernel<<<>>> doesn't return anything, we need another way to get the error code
#define DebugKernel \
    cudaDeviceSynchronize(); \
    printf("Kernel launch: %s\n", cudaGetErrorString(cudaGetLastError()));

    // Utility for setting blockDim and gridDim (1D). A block cannot have more than 1024 threads
    // number of threads needed, 2 output params
    inline void setKernelDim1D(int threads, dim3& gridDim, dim3& blockDim)
    {
        if (threads > 1024) // we don't have enough threads on a single block
        {
            gridDim.x = threads / 1024 + 1;
            blockDim.x = 1024;
        }
        else // try to make block dim a multiple of 32 to conform with 'warp'
        {
            if (threads % 32 == 0) blockDim.x = threads;
            else blockDim.x = (threads / 32 + 1) * 32;
        }
    }

    // The specified col will be set to a specific value
    // negative 'colIdx' means counting from the last col (-n => col - n)
	template <typename T>
    inline void gpu_fill_col(device_ptr<T> begin, int row, int col, int colIdx, T val)
    {
        if (colIdx < 0)  colIdx += col;
        thrust::fill_n(begin + row * colIdx, row, val);
    }

    // Change a specific row of a column major matrix to a specific value
    // negative 'rowIdx' means counting from the last row (-n => row - n)
	template <typename T>
    __global__
	void kernel_fill_row(T *begin, int row, int col, int rowIdx, T val)
	{
		ThreadIndex1D(idx, col);

		begin += row * idx + rowIdx; // end of a column
		*begin = val; // set the value
	}

    // The specified row will be set to a specific value
    // negative 'rowIdx' means counting from the last row (-n => row - n)
	template <typename T>
    inline void gpu_fill_row(device_ptr<T> begin, int row, int col, int rowIdx, T val)
    {
        dim3 gridDim, blockDim;
        setKernelDim1D(col, gridDim, blockDim);

        if (rowIdx < 0) rowIdx += row;

		kernel_fill_row<T> << <gridDim, blockDim >> >(
                thrust::raw_pointer_cast(begin), row, col, rowIdx, val);
    }


    // Matrix transposition
    // Code from http://www.evl.uic.edu/aej/525/code/transpose_kernel.cu
    // http://www.evl.uic.edu/aej/525/code/transpose.cu
#define BLOCK_DIM 16
	template <typename T>
    __global__ void kernel_transpose(T *in, int width, int height, T *out)
    {
        __shared__ T block[BLOCK_DIM][BLOCK_DIM + 1];

        // read the matrix tile into shared memory
        unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
        unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
        if ((xIndex < width) && (yIndex < height))
        {
            unsigned int index_in = yIndex * width + xIndex;
            block[threadIdx.y][threadIdx.x] = in[index_in];
        }

        __syncthreads();

        // write the transposed matrix tile to global memory
        xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
        yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
        if ((xIndex < height) && (yIndex < width))
        {
            unsigned int index_out = yIndex * height + xIndex;
            out[index_out] = block[threadIdx.x][threadIdx.y];
        }
    }

    // Transposes 'in' and fills 'out'
	template <typename T>
    inline void gpu_transpose(device_ptr<T> in, int row, int col, device_ptr<T> out)
    {
        dim3 gridDim(std::ceil(1.0 *row / BLOCK_DIM), std::ceil(1.0 * col / BLOCK_DIM)),
             blockDim(BLOCK_DIM, BLOCK_DIM);

		kernel_transpose<T> << <gridDim, blockDim >> >(
                thrust::raw_pointer_cast(in), row, col, thrust::raw_pointer_cast(out));
    }

    /**********************************************
    * Softmax/labeling related kernels  *
	All __device__ functions should be prefixed with 'device_'
	All __global__ kernel<<< >>> functions should be prefixed with 'kernel_'
    **********************************************/
	template <typename T>
    __inline__ __device__
	T device_max(T *begin, int size)
	{
		// find max
		T mx = -1e20;
		for (int i = 0; i < size; i++)
			if (begin[i] > mx) mx = begin[i];
		return mx;
	}

	template <typename T>
	__inline__ __device__
	void device_softmax(T *begin, int row, int col, T *out)
	{
		// find max
		T mx = device_max<T>(begin, row);
		// subtract max from each and do exp
		// also compute sum of these exp
		T sum = 0;
		for (int i = 0; i < row; i++)
		{
			out[i] = exp(begin[i] - mx);
			sum += out[i];
		}
		// exp(..)/sum
		for (int i = 0; i < row; i++)
			out[i] /= sum;
	}

	///// mini-batch softmax(alpha_vec)
	template <typename T>
	__global__
	void kernel_batch_softmax(T *begin, int row, int col, T *out)
	{
		ThreadIndex1D(idx, col);
		begin += idx * row; // beginning of a column
		if (out != begin) out += idx * row;
		device_softmax<T>(begin, row, col, out);
	}

	// Computes the mini-batch softmax probability distribution for the full matrix
	template <typename T>
	inline void gpu_batch_softmax(
		device_ptr<T> begin, int row, int col, device_ptr<T> out)
	{
		if (col == 1) // use the thrust version
			gpu_softmax<T>(begin, row, out);
		else // real batch
		{
			dim3 gridDim, blockDim;
			setKernelDim1D(col, gridDim, blockDim);

			kernel_batch_softmax<T> << <gridDim, blockDim >> >(
				thrust::raw_pointer_cast(begin), row, col, thrust::raw_pointer_cast(out));
		}
	}

	template <typename T>
	inline void gpu_batch_softmax(
		device_ptr<T> begin, int row, int col)
	{
		gpu_batch_softmax<T>(begin, row, col, begin);
	}

	///// mini-batch softmax(alpha_vec) that writes to 'outProb' only the probability at the correct label
	// Non-intrusive: doesn't change 'begin'
	template <typename T>
	__global__
	void kernel_batch_softmax_at_label(
		T *begin, int row, int col, T *outProb, int *labels)
	{
		ThreadIndex1D(idx, col);

		begin += idx * row; // beginning of a column

		// find max
		T mx = device_max<T>(begin, row);
		// subtract max from each and do exp
		// also compute sum of these exp
		T sum = 0;
		T numerator;
		int label = labels[idx];
		for (int i = 0; i < row; i++)
		{
			if (i == label) numerator = begin[i] - mx;
			sum += exp(begin[i] - mx);
		}
		outProb[idx] = numerator - log(sum);
	}

	// writes to 'outLogProb' only the log(probability) at the correct label
	// return sum(outLogProb)
	// int *labels is on GPU
	// input data 'begin' won't be changed
	template <typename T>
	inline float gpu_batch_softmax_at_label(
		device_ptr<T> begin, int row, int col, device_ptr<T> outLogProb, int *labels)
	{
		if (col == 1) // use the thrust version
		{
			int label;
			cudaMemcpy(&label, labels, sizeof(int), cudaMemcpyDeviceToHost);
			return gpu_softmax_at_label<T>(begin, row, label, outLogProb);
		}
		else // real batch
		{
			dim3 gridDim, blockDim;
			setKernelDim1D(col, gridDim, blockDim);

			kernel_batch_softmax_at_label<T> << <gridDim, blockDim >> >(
				thrust::raw_pointer_cast(begin), row, col, thrust::raw_pointer_cast(outLogProb), labels);

			return gpu_sum<T>(outLogProb, col);
		}
	}

    // Used in 2 other kernels
    // return the probability (before log()) at the correct label
	template <typename T>
    __inline__ __device__
	T kernel_softmax_minus_id(int idx, T *begin, int row, int col, T *out, int *labels)
	{
		begin += idx * row; // beginning of a column
		if (out != begin) out += idx * row;

		device_softmax<T>(begin, row, col, out);

		// Add 1 to the identity function
		T probCorrect = out[labels[idx]];
		-- out[labels[idx]];
		return probCorrect;
	}

    ///// mini-batch softmax(alpha_vec) - I [y==j]
	template <typename T>
    __global__
	void kernel_batch_softmax_minus_id(
			T *begin, int row, int col, T *out, int *labels)
	{
		ThreadIndex1D(idx, col);

		kernel_softmax_minus_id<T>(idx, begin, row, col, out, labels);
	}

	template <typename T>
    inline void gpu_batch_softmax_minus_id(
            device_ptr<T> begin, int row, int col, device_ptr<T> out, int *labels)
    {
        if (col == 1) // use the thrust version
        {
            int label;
            cudaMemcpy(&label, labels, sizeof(int), cudaMemcpyDeviceToHost);
			gpu_softmax_minus_id<T>(begin, row, out, label);
        }
        else // real batch
        {
            dim3 gridDim, blockDim;
            setKernelDim1D(col, gridDim, blockDim);

			kernel_batch_softmax_minus_id<T> << <gridDim, blockDim >> >(
				thrust::raw_pointer_cast(begin), row, col, thrust::raw_pointer_cast(out), labels);
        }
    }
	// Overload: in == out
	template <typename T>
	inline void gpu_batch_softmax_minus_id(
		device_ptr<T> begin, int row, int col, int *labels)
	{
		gpu_batch_softmax_minus_id<T>(begin, row, col, begin, labels);
	}

	///// Fill 'outLabels' with the label corresponding to the maximum probability of a column
	template <typename T>
	__global__
	void kernel_best_label(
			T *begin, int row, int col, int *outLabels)
	{
		ThreadIndex1D(idx, col);

		begin += idx * row; // beginning of a column

		// find max
		T mx = -1e20;
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
	template <typename T>
    inline void gpu_best_label(
            device_ptr<T> begin, int row, int col, int *outLabels)
    {
        dim3 gridDim, blockDim;
        setKernelDim1D(col, gridDim, blockDim);

		kernel_best_label<T> << <gridDim, blockDim >> >(
                thrust::raw_pointer_cast(begin), row, col, outLabels);
    }
}

#endif // my_kernel_h__
