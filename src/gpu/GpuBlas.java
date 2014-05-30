package gpu;

import utils.GpuUtil;
import utils.PP;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import static jcuda.jcublas.cublasOperation.*;
import static jcuda.Sizeof.*;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.runtime.JCuda.*;

/**
 * JCublas context
 */
public class GpuBlas
{
	// Cublas context
	private static cublasHandle handle = null;

	/**
	 * Initialize cublas context
	 */
	public static void init()
	{
		if (handle == null)
		{
			handle = new cublasHandle();
			cublasCreate(handle);
		}
	}

	/**
	 * Destroy cublas context
	 */
	public static void destroy()
	{
		cublasDestroy(handle);
		handle = null;
	}


	/**
	 * Multiply two FloatMat and add onto an existing FloatMat.
	 * The most complete method. All others overload from this.
	 * C = alpha * A * B + beta * C;
	 * @return input parameter C
	 */
	public static FloatMat mult(FloatMat A, FloatMat B, FloatMat C, float alpha, float beta)
	{
		return GpuBlas.mult(A, B, C, alpha, beta, 
				A.numRows, A.numCols, B.numCols
			);
	}
	
	// DIMENSIONS OF ARRAYS TO MULTIPLY
	// A: m x k
	// B: k x n
	// C: m x n
	// m, n, k are named according to the online documentation
	// http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
	// Note that m,n,k might specify a sub-matrix of A, B, or C
	// to use in this multiplication. (it is not necessarily true that
	// m = A.numRows, k = A.numCols (or B.numRows), n = B.numCols)
	public static FloatMat mult(
			FloatMat A, FloatMat B, FloatMat C, float alpha, float beta,
			int m, int k, int n)
	{
		Pointer pa = A.getDevicePointer();
		Pointer pb = B.getDevicePointer();
		Pointer pc = C.getDevicePointer();
		
		cublasSgemm(handle, A.getOp(), B.getOp(), 
				m, n, k, 
				GpuUtil.toFloatPointer(alpha), 
				pa, A.ldim, 
				pb, B.ldim, 
				GpuUtil.toFloatPointer(beta), 
				pc, C.ldim);

		return C;
	}
	

	/**
	 * Multiply two FloatMat and add onto an existing FloatMat.
	 * C = A * B;
	 * @return input parameter C
	 */
	public static FloatMat mult(FloatMat A, FloatMat B, FloatMat C)
	{
		return GpuBlas.mult(A, B, C, 1, 0);
	}

	/**
	 * Multiply two FloatMat
	 * @return C = alpha * A *B
	 * @throws GpuException 
	 */
	public static FloatMat mult(FloatMat A, FloatMat B, float alpha) throws GpuException
	{
		return GpuBlas.mult(
				A, B, 
				new FloatMat(A.numRows, B.numCols, false /*memsetToZero*/), 
				alpha, 0
			);
	}

	/**
	 * Multiply two FloatMat
	 * @return C = A * B
	 * @throws GpuException 
	 */
	public static FloatMat mult(FloatMat A, FloatMat B) throws GpuException
	{	
		return GpuBlas.mult(A, B, 1);
	}
	
	/**
	 * Matrix multiplies vector and add onto an existing vector (col=1)
	 * The most complete method. All others overload from this.
	 * y = alpha * A * x + beta * y
	 * @return input parameter y
	 */
	public static FloatMat multVec(FloatMat A, FloatMat x, FloatMat y, float alpha, float beta)
	{
		Pointer pa = A.getDevicePointer();
		Pointer px = x.getDevicePointer();
		Pointer py = y.getDevicePointer();
		// Here is an inconsistency in the API
		// m and n are the original row/col dimension
		int m = A.getOriginalNumRows();
		int n = A.getOriginalNumCols();
		
		cublasSgemv(handle, A.getOp(),
				m, n, 
				GpuUtil.toFloatPointer(alpha), pa, A.ldim, 
				px, 1, 
				GpuUtil.toFloatPointer(beta), py, 1);

		return y;
	}
	
	/**
	 * Matrix multiplies vector and add onto an existing vector (col=1)
	 * y = A * x
	 * @return input parameter y
	 */
	public static FloatMat multVec(FloatMat A, FloatMat x, FloatMat y)
	{
		return GpuBlas.multVec(A, x, y, 1, 0);
	}
	
	/**
	 * Matrix multiplies vector
	 * @return y = alpha * A * x + beta * y
	 * @throws GpuException 
	 */
	public static FloatMat multVec(FloatMat A, FloatMat x, float alpha, float beta) throws GpuException
	{
		return GpuBlas.multVec(
				A, x, 
				new FloatMat(A.numRows, 1, false /*memsetToZero*/), 
				alpha, beta
			);
	}
	
	/**
	 * Matrix multiplies vector
	 * @return y = A * x
	 * @throws GpuException 
	 */
	public static FloatMat multVec(FloatMat A, FloatMat x) throws GpuException
	{
		return GpuBlas.multVec(
				A, x, 
				new FloatMat(A.numRows, 1, false /*memsetToZero*/), 
				1, 0
			);
	}
	

	/**
	 * Add two FloatMat.
	 * The most complete method. All others overload from this. 
	 * C = alpha * A + beta * B;
	 * @return input parameter C
	 */
	public static FloatMat add(FloatMat A, FloatMat B, FloatMat C, float alpha, float beta)
	{
		Pointer pa = A.getDevicePointer();
		Pointer pb = B.getDevicePointer();
		Pointer pc = C.getDevicePointer();
		int m = A.numRows;
		int n = A.numCols;

		cublasSgeam(handle, A.getOp(), B.getOp(), 
				m, n, 
				GpuUtil.toFloatPointer(alpha), pa, A.ldim, 
				GpuUtil.toFloatPointer(beta), pb, B.ldim, 
				pc, C.ldim);

		return C;
	}

	/**
	 * Add two FloatMat
	 * C = A + B
	 * @return input parameter C
	 */
	public static FloatMat add(FloatMat A, FloatMat B, FloatMat C)
	{
		return GpuBlas.add(A, B, C, 1, 1);
	}
	
	/**
	 * Add two FloatMat
	 * @return C = alpha * A + beta * B
	 * @throws GpuException 
	 */
	public static FloatMat add(FloatMat A, FloatMat B, float alpha, float beta) throws GpuException
	{
		return GpuBlas.add(
				A, B, 
				new FloatMat(A.numRows, A.numCols, false /*memsetToZero*/), 
				alpha, beta
			);
	}
	/**
	 * Add two FloatMat
	 * @return C = A + B
	 * @throws GpuException 
	 */
	public static FloatMat add(FloatMat A, FloatMat B) throws GpuException
	{
		return GpuBlas.add(A, B, 1, 1);
	}
	
	/**
	 * Copies a matrix device data to another. They should have the same dim. 
	 * @return input parameter 'to'
	 */
	public static FloatMat copy(FloatMat from, FloatMat to)
	{
		cublasScopy(handle, from.size(), 
				from.getDevicePointer(), 1, 
				to.getDevicePointer(), 1);
		return to;
	}
	
	/**
	 * Copies a matrix device data to another. 
	 * @return the new clone
	 * @throws GpuException 
	 */
	public static FloatMat copy(FloatMat from) throws GpuException
	{
		return GpuBlas.copy(
				from, 
				new FloatMat(from.numRows, from.numCols, false /*memsetToZero*/)
			);
	}
	
	/**
	 * @return the index of the maximum absolute value
	 * NOTE: absolute value, no sign
	 */
	public static int maxAbsIndex(FloatMat A)
	{
		int[] hostIdx = new int[1];
		Pointer deviceIdx = Pointer.to(hostIdx);
		cublasIsamax(handle, A.size(), A.getDevicePointer(), 1, deviceIdx);
		cudaFree(deviceIdx);
		return hostIdx[0] - 1; // adjust to 0-based
	}
	
	/**
	 * @return the index of the minimum absolute value
	 * NOTE: absolute value, no sign
	 */
	public static int minAbsIndex(FloatMat A)
	{
		int[] idx = new int[1];
		Pointer idxPtr = Pointer.to(idx);
		cublasIsamin(handle, A.size(), A.getDevicePointer(), 1, idxPtr);
		return idx[0] - 1; // adjust to 0-based
	}
	
	/**
	 * @return L2-norm of a vector
	 */
	public static float norm(FloatMat A)
	{
		float[] val = new float[1];
		Pointer valPtr = Pointer.to(val);
		cublasSnrm2(handle, A.size(), A.getDevicePointer(), 1, valPtr);
		return val[0];
	}
	
	/**
	 * Scale and add onto an existing matrix
	 * y = alpha * x + y
	 * @return input parameter y
	 */
	public static FloatMat scaleAdd(FloatMat x, FloatMat y, float alpha)
	{
		cublasSaxpy(handle, x.size(), 
				GpuUtil.toFloatPointer(alpha), 
				x.getDevicePointer(), 1, 
				y.getDevicePointer(), 1);
		return y;
	}
	
	/**
	 * Scale and add onto an existing matrix
	 * y += x
	 * @return input parameter y
	 */
	public static FloatMat scaleAdd(FloatMat x, FloatMat y)
	{	
		return scaleAdd(x, y, 1);
	}
	
	/**
	 * Scale and overwrite an existing matrix
	 * x *= alpha
	 * @return input parameter x
	 */
	public static FloatMat scale(FloatMat x, float alpha)
	{
		cublasSscal(handle, x.size(), 
				GpuUtil.toFloatPointer(alpha), 
				x.getDevicePointer(), 1);
		return x;
	}
	
	/**
	 * @return dot product of vectors x and y
	 */
	public static float dot(FloatMat x, FloatMat y)
	{
		float[] val = new float[1];
		Pointer valPtr = Pointer.to(val);
		cublasSdot(handle, x.size(), 
				x.getDevicePointer(), 1, 
				y.getDevicePointer(), 1, 
				valPtr);
		return val[0];
	}


	/**
	 * Create and copy to Cublas device vector
	 * @return new device pointer
	 * @throws GpuException 
	 */
	public static Pointer hostToCublasFloat(float[] host) throws GpuException
	{
		int n = host.length;
		Pointer device = GpuUtil.allocateDeviceFloat(n);
		cublasSetVector(n, FLOAT, 
				Pointer.to(host), 1, 
				device, 1);
		return device;
	}
	
	
	// Copies n elements from a vector x in CPU memory space to a vector y in GPU memory space. 
	// Elements in both vectors are assumed to have a size of elemSize bytes. 
	// Storage spacing between consecutive elements is incx for the source vector x and 
	// incy for the destination vector y. In general, y points to an object, or part of an object,
	// allocated via cublasAlloc(). Column major format for two-dimensional matrices is assumed 
	// throughout CUBLAS. Therefore, if the increment for a vector is equal to 1, this access a 
	// column vector while using an increment equal to the leading dimension of the respective
	// matrix accesses a row vector.
	public static void setFloatVectorDevice(
			Pointer hostPtr, int incHost, Pointer devicePtr, int incDevice, int numElementsToCopy)
	{
		cublasSetVector(numElementsToCopy, Sizeof.FLOAT /*elemSize*/, hostPtr, incHost, devicePtr, incDevice);
	}
	
	/**
	 * Create and copy to Cublas device vector
	 */
	public static void hostToCublasFloat(float[] host, Pointer device)
	{
		cublasSetVector(host.length, FLOAT, 
				Pointer.to(host), 1, 
				device, 1);
	}
	
	/**
	 * Copy the device vector at Cublas back to host
	 * @return new host array
	 */
	public static float[] cublasToHostFloat(Pointer device, int size)
	{
		float[] host = new float[size];
		cublasGetVector(size, FLOAT, device, 1, Pointer.to(host), 1);
		return host;
	}

	/**
	 * Copy the device vector at Cublas back to host
	 */
	public static void cublasToHostFloat(Pointer device, float[] host)
	{
		cublasGetVector(host.length, FLOAT, device, 1, Pointer.to(host), 1);
	}
}
