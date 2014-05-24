package gpu;

import utils.GpuUtil;
import utils.PP;
import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;
import static jcuda.jcublas.cublasOperation.*;
import static jcuda.Sizeof.*;
import static jcuda.jcublas.JCublas2.*;

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
				m, n, k, GpuUtil.toPointer(alpha), 
				pa, A.ldim, pb, B.ldim, 
				GpuUtil.toPointer(beta), pc, C.ldim);

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
				GpuUtil.toPointer(alpha), pa, A.ldim, 
				px, 1, 
				GpuUtil.toPointer(beta), py, 1);

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
				GpuUtil.toPointer(alpha), pa, A.ldim, 
				GpuUtil.toPointer(beta), pb, B.ldim, 
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
		return add(A, B, new FloatMat(A.row, A.col), alpha, beta);
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
	 * Create and copy to Cublas device vector
	 */
	public static Pointer toCublasFloat(float[] host)
	{
		int n = host.length;
		Pointer p = GpuUtil.createDeviceFloat(n);

		cublasSetVector(n, FLOAT, Pointer.to(host), 1, p, 1);

		return p;
	}

	/**
	 * Copy the device vector at Cublas back to host
	 */
	public static void toHostFloat(Pointer device, float[] host)
	{
		cublasGetVector(host.length, FLOAT, device, 1, Pointer.to(host), 1);
	}
}
