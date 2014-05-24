package gpu;

import utils.GpuUtil;
import utils.PP;
import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import static jcuda.Sizeof.*;
import static jcuda.jcublas.JCublas2.*;

/**
 * JCublas context
 */
public class GpuBlas
{
	// Cublas context
	private static cublasHandle handle = null;
	private static final int NOP = cublasOperation.CUBLAS_OP_N;
	
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
	 * Multiply two FloatMat
	 * @return C = alpha * A *B
	 */
	public static FloatMat mult(FloatMat A, FloatMat B, float alpha)
	{
		Pointer pa = A.getDevice();
		Pointer pb = B.getDevice();
		// m, n, k are named according to the online documentation
		// http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm
		int m = A.row;
		int k = A.col;  // = B.row
		int n = B.col;

		/* leading dimension args:
		int lda = m; // = A's column length, which is row dim
		int ldb = k; // = B's column length
		int ldc = m; // = C's column length */		

		// Store the result to C. Init C's memory to 0. 
		Pointer pc = GpuUtil.createDeviceFloat(m * n, true);
		cublasSgemm(handle, NOP, NOP, 
				m, n, k, GpuUtil.toPointer(alpha), 
				pa, m, pb, k, 
				GpuUtil.toPointer(0), pc, m);
		
		return new FloatMat(pc, m, n);
	}
	
	/**
	 * Multiply two FloatMat
	 * @return C = A * B
	 */
	public static FloatMat mult(FloatMat A, FloatMat B) {	return mult(A, B, 1); }
	
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

		/* leading dimension args:
		int lda = m; // = A's column length, which is row dim
		int ldb = k; // = B's column length
		int ldc = m; // = C's column length */		

		// Store the result to C. Init C's memory to 0. 
		cublasSgemm(handle, NOP, NOP, 
				m, n, k, GpuUtil.toPointer(alpha), 
				pa, m, pb, k, 
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
