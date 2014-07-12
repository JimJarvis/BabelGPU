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

        gpu_fill_row_float_kernel << <gridDim, blockDim >> >(
                thrust::raw_pointer_cast(begin), row, col, rowIdx, val);
    }


    // Matrix transposition
    // Code from http://www.evl.uic.edu/aej/525/code/transpose_kernel.cu
    // http://www.evl.uic.edu/aej/525/code/transpose.cu
#define BLOCK_DIM 16
    __global__ void gpu_transpose_float_kernel(float *in, int width, int height, float *out)
    {
        __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

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
    inline void gpu_transpose_float(device_ptr<float> in, int row, int col, device_ptr<float> out)
    {
        dim3 gridDim(std::ceil(1.0 *row / BLOCK_DIM), std::ceil(1.0 * col / BLOCK_DIM)),
             blockDim(BLOCK_DIM, BLOCK_DIM);

        gpu_transpose_float_kernel << <gridDim, blockDim >> >(
                thrust::raw_pointer_cast(in), row, col, thrust::raw_pointer_cast(out));
    }

    /**********************************************/
    /* Common device functions used in kernels  */
    /**********************************************/
    __inline__ __device__
	float max_kernel(float *begin, int size)
	{
		// find max
		float mx = -1e20;
		for (int i = 0; i < size; i++)
			if (begin[i] > mx) mx = begin[i];
		return mx;
	}

    // Used in 2 other kernels
    // return the probability (before log()) at the correct label
    __inline__ __device__
	float id_minus_softmax_kernel(int idx, float *begin, int row, int col, int *labels)
	{
		begin += idx * row; // beginning of a column

		// find max
		float mx = max_kernel(begin, row);
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
		float probCorrect = -begin[labels[idx]];
		++begin[labels[idx]];
		return probCorrect;
	}

    ///// mini-batch I [y==j] - softmax(alpha_vec)
    __global__
	void gpu_batch_id_minus_softmax_kernel(
			float *begin, int row, int col, int *labels)
	{
		ThreadIndex1D(idx, col);

		id_minus_softmax_kernel(idx, begin, row, col, labels);
	}

    inline void gpu_batch_id_minus_softmax(
            device_ptr<float> begin, int row, int col, int *labels)
    {
        if (col == 1) // use the thrust version
        {
            int label;
            cudaMemcpy(&label, labels, sizeof(int), cudaMemcpyDeviceToHost);
            gpu_id_minus_softmax(begin, row, label);
        }
        else // real batch
        {
            dim3 gridDim, blockDim;
            setKernelDim1D(col, gridDim, blockDim);

            gpu_batch_id_minus_softmax_kernel << <gridDim, blockDim >> >(
                    thrust::raw_pointer_cast(begin), row, col, labels);
        }
    }

    ///// id-softmax AND return the sum of log probability.
    // combine gpu_batch_id_minus_softmax and gpu_log_prob
    __global__
	void gpu_batch_id_minus_softmax_log_prob_kernel(
			float *begin, int row, int col, float *outLogProb, int *labels)
	{
		ThreadIndex1D(idx, col);

		outLogProb[idx] = log(
				id_minus_softmax_kernel(idx, begin, row, col, labels));
	}

    // Computes the id - softmax() for each column, and return the sum of the log prob at the correct labels
    // writes the log probability at the correct labels to 'outLogProb'
    inline float gpu_batch_id_minus_softmax_log_prob(
            device_ptr<float> begin, int row, int col, device_ptr<float> outLogProb, int *labels)
    {
        dim3 gridDim, blockDim;
        setKernelDim1D(col, gridDim, blockDim);

        gpu_batch_id_minus_softmax_log_prob_kernel << <gridDim, blockDim >> >(
                thrust::raw_pointer_cast(begin), row, col, thrust::raw_pointer_cast(outLogProb), labels);

        return gpu_sum_float(outLogProb, col);
    }

    ///// mini-batch softmax(alpha_vec)
    __global__
	void gpu_batch_softmax_kernel(float *begin, int row, int col, float *out)
	{
		ThreadIndex1D(idx, col);

		begin += idx * row; // beginning of a column
		if (out != begin) out += idx * row;

		// find max
		float mx = max_kernel(begin, row);
		// subtract max from each and do exp
		// also compute sum of these exp
		float sum = 0;
		for (int i = 0; i < row; i++)
		{
			out[i] = exp(begin[i] - mx);
			sum += out[i];
		}
		// exp(..)/sum
		for (int i = 0; i < row; i++)
			out[i] /= sum;
	}

    // Computes the mini-batch softmax probability distribution for the full matrix
    inline void gpu_batch_softmax(
            device_ptr<float> begin, int row, int col, device_ptr<float> out)
    {
        if (col == 1) // use the thrust version
            gpu_softmax(begin, row);
        else // real batch
        {
            dim3 gridDim, blockDim;
            setKernelDim1D(col, gridDim, blockDim);

            gpu_batch_softmax_kernel << <gridDim, blockDim >> >(
                    thrust::raw_pointer_cast(begin), row, col, thrust::raw_pointer_cast(out));
        }
    }

	inline void gpu_batch_softmax(
		device_ptr<float> begin, int row, int col)
	{
		gpu_batch_softmax(begin, row, col, begin);
	}

    ///// mini-batch softmax(alpha_vec) that writes to 'out' only the probability at the correct label
    __global__
	void gpu_batch_softmax_kernel(
			float *begin, int row, int col, float *outProb, int *labels)
	{
		ThreadIndex1D(idx, col);

		begin += idx * row; // beginning of a column

		// find max
		float mx = max_kernel(begin, row);
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
		outProb[idx] = numerator / sum;
	}

    // Computes the mini-batch softmax probability distribution for the full matrix
    // writes to 'out' only the probability at the correct label
    // int *labels is on GPU
    inline void gpu_batch_softmax(
            device_ptr<float> begin, int row, int col, device_ptr<float> outProb, int *labels)
    {
        if (col == 1) // use the thrust version
        {
            int label;
            cudaMemcpy(&label, labels, sizeof(int), cudaMemcpyDeviceToHost);
            gpu_softmax(begin, row, label, outProb);
        }
        else // real batch
        {
            dim3 gridDim, blockDim;
            setKernelDim1D(col, gridDim, blockDim);

            gpu_batch_softmax_kernel << <gridDim, blockDim >> >(
                    thrust::raw_pointer_cast(begin), row, col, thrust::raw_pointer_cast(outProb), labels);
        }
    }

    ///// Fill 'outLabels' with the label corresponding to the maximum probability of a column
    __global__
	void gpu_best_label_kernel(
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
    inline void gpu_best_label(
            device_ptr<float> begin, int row, int col, int *outLabels)
    {
        dim3 gridDim, blockDim;
        setKernelDim1D(col, gridDim, blockDim);

        gpu_best_label_kernel << <gridDim, blockDim >> >(
                thrust::raw_pointer_cast(begin), row, col, outLabels);
    }
}

#endif // my_kernel_h__
