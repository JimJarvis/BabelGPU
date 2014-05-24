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
	 * Multiply two FloatMat and add onto an existing FloatMat
	 * C = alpha * A * B + beta * C;
	 * @return input parameter C
	 */
	public static FloatMat mult(FloatMat A, FloatMat B, FloatMat C, float alpha, float beta)
	{
		Pointer pa = A.getDevice();
		Pointer pb = B.getDevice();
		// m, n, k are named according to the online documentation
		// http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
		int m = A.row;
		int k = A.col;  // = B.row
		int n = B.col;
		
		// int ldc = m; // = C's column length	

		// Store the result to C. Init C's memory to 0. 
		cublasSgemm(handle, A.getOp(), B.getOp(), 
				m, n, k, GpuUtil.toPointer(alpha), 
				pa, A.ldim, pb, B.ldim, 
				GpuUtil.toPointer(beta), C.getDevice(), m);
		
		return C;
	}
	
	/**
	 * Multiply two FloatMat and add onto an existing FloatMat
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
	public static FloatMat mult(FloatMat A, FloatMat B) {	return mult(A, B, 1); }
	
	/**
	 * Add two FloatMat
	 * @return C = alpha * A + beta * B;
	 */
	 public static FloatMat add(FloatMat A, FloatMat B)
	 {
		 return null;
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
