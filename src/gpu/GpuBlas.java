package gpu;

import utils.GpuUtil;
import utils.PP;
import jcuda.Pointer;
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
		Pointer pa = A.getDevice();
		Pointer pb = B.getDevice();
		Pointer pc = C.getDevice();
		// m, n, k are named according to the online documentation
		// http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
		int m = A.row;
		int k = A.col;  // = B.row
		int n = B.col;

		cublasSgemm(handle, A.getOp(), B.getOp(), 
				m, n, k, GpuUtil.toHostFloatPointer(alpha), 
				pa, A.ldim, pb, B.ldim, 
				GpuUtil.toHostFloatPointer(beta), pc, C.ldim);

		return C;
	}

	/**
	 * Multiply two FloatMat and add onto an existing FloatMat.
	 * C = A * B;
	 * @return input parameter C
	 */
	public static FloatMat mult(FloatMat A, FloatMat B, FloatMat C)
	{
		return mult(A, B, C, 1, 0);
	}

	/**
	 * Multiply two FloatMat
	 * @return C = alpha * A *B
	 */
	public static FloatMat mult(FloatMat A, FloatMat B, float alpha)
	{
		return mult(A, B, new FloatMat(A.row, B.col), alpha, 0);
	}

	/**
	 * Multiply two FloatMat
	 * @return C = A * B
	 */
	public static FloatMat mult(FloatMat A, FloatMat B)
	{	
		return mult(A, B, 1);
	}
	
	/**
	 * Matrix multiplies vector and add onto an existing vector (col=1)
	 * The most complete method. All others overload from this.
	 * y = alpha * A * x + beta * y
	 * @return input parameter y
	 */
	public static FloatMat multVec(FloatMat A, FloatMat x, FloatMat y, float alpha, float beta)
	{
		Pointer pa = A.getDevice();
		Pointer px = x.getDevice();
		Pointer py = y.getDevice();
		// Here is an inconsistency in the API
		// m and n are the original row/col dimension
		int m = A.getOriginalRow();
		int n = A.getOriginalCol();
		
		cublasSgemv(handle, A.getOp(),
				m, n, 
				GpuUtil.toHostFloatPointer(alpha), pa, A.ldim, 
				px, 1, 
				GpuUtil.toHostFloatPointer(beta), py, 1);

		return y;
	}
	
	/**
	 * Matrix multiplies vector and add onto an existing vector (col=1)
	 * y = A * x
	 * @return input parameter y
	 */
	public static FloatMat multVec(FloatMat A, FloatMat x, FloatMat y)
	{
		return multVec(A, x, y, 1, 0);
	}
	
	/**
	 * Matrix multiplies vector
	 * @return y = alpha * A * x + beta * y
	 */
	public static FloatMat multVec(FloatMat A, FloatMat x, float alpha, float beta)
	{
		return multVec(A, x, new FloatMat(A.row, 1), alpha, beta);
	}
	
	/**
	 * Matrix multiplies vector
	 * @return y = A * x
	 */
	public static FloatMat multVec(FloatMat A, FloatMat x)
	{
		return multVec(A, x, new FloatMat(A.row, 1), 1, 0);
	}
	

	/**
	 * Add two FloatMat.
	 * The most complete method. All others overload from this. 
	 * C = alpha * A + beta * B;
	 * @return input parameter C
	 */
	public static FloatMat add(FloatMat A, FloatMat B, FloatMat C, float alpha, float beta)
	{
		Pointer pa = A.getDevice();
		Pointer pb = B.getDevice();
		Pointer pc = C.getDevice();
		int m = A.row;
		int n = A.col;

		cublasSgeam(handle, A.getOp(), B.getOp(), 
				m, n, 
				GpuUtil.toHostFloatPointer(alpha), pa, A.ldim, 
				GpuUtil.toHostFloatPointer(beta), pb, B.ldim, 
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
		return add(A, B, C, 1, 1);
	}
	
	/**
	 * Add two FloatMat
	 * @return C = alpha * A + beta * B
	 */
	public static FloatMat add(FloatMat A, FloatMat B, float alpha, float beta)
	{
		return add(A, B, new FloatMat(A.row, A.col, false), alpha, beta);
	}
	/**
	 * Add two FloatMat
	 * @return C = A + B
	 */
	public static FloatMat add(FloatMat A, FloatMat B)
	{
		return add(A, B, 1, 1);
	}
	
	/**
	 * Copies a matrix device data to another. They should have the same dim. 
	 * @return input parameter 'to'
	 */
	public static FloatMat copy(FloatMat from, FloatMat to)
	{
		cublasScopy(handle, from.size(), 
				from.getDevice(), 1, 
				to.getDevice(), 1);
		return to;
	}
	
	/**
	 * Copies a matrix device data to another. 
	 * @return the new clone
	 */
	public static FloatMat copy(FloatMat from)
	{
		return copy(from, new FloatMat(from.row, from.col, false));
	}
	
	/**
	 * @return the index of the maximum absolute value
	 * NOTE: absolute value, no sign
	 */
	public static int maxAbsIndex(FloatMat A)
	{
		int[] hostIdx = new int[1];
		Pointer deviceIdx = Pointer.to(hostIdx);
		cublasIsamax(handle, A.size(), A.getDevice(), 1, deviceIdx);
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
		cublasIsamin(handle, A.size(), A.getDevice(), 1, idxPtr);
		return idx[0] - 1; // adjust to 0-based
	}
	
	/**
	 * @return L2-norm of a vector
	 */
	public static float norm(FloatMat A)
	{
		float[] val = new float[1];
		Pointer valPtr = Pointer.to(val);
		cublasSnrm2(handle, A.size(), A.getDevice(), 1, valPtr);
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
				GpuUtil.toHostFloatPointer(alpha), 
				x.getDevice(), 1, 
				y.getDevice(), 1);
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
				GpuUtil.toHostFloatPointer(alpha), 
				x.getDevice(), 1);
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
				x.getDevice(), 1, 
				y.getDevice(), 1, 
				valPtr);
		return val[0];
	}
	
	/**
	 * Swap two FloatMat's device data
	 */
	public static void swap(FloatMat x, FloatMat y)
	{
		cublasSswap(handle, x.size(), 
				x.getDevice(), 1, 
				y.getDevice(), 1);
	}


	/**
	 * Create and copy to Cublas device vector
	 */
	public static Pointer hostToCublasFloat(float[] host)
	{
		int n = host.length;
		Pointer p = GpuUtil.createDeviceFloat(n);

		cublasSetVector(n, FLOAT, Pointer.to(host), 1, p, 1);

		return p;
	}

	/**
	 * Copy the device vector at Cublas back to host
	 */
	public static void cublasToHostFloat(Pointer device, float[] host)
	{
		cublasGetVector(host.length, FLOAT, device, 1, Pointer.to(host), 1);
	}
}
