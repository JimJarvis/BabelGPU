package gpu;

import utils.GpuUtil;
import utils.PP;
import jcuda.Pointer;
import static jcuda.runtime.JCuda.*;
import static jcuda.jcublas.cublasOperation.*;

/**
 * Struct around a matrix with row/col dimension info
 */
public class FloatMat
{
	private float[] host = null;
	private Pointer device = null;
	// This field records whether the matrix should be transposed or not
	private int op = CUBLAS_OP_N; 
	public int row;
	public int col;
	// Leading dimension: column length (i.e. row dim)
	// Doesn't change even with transpose
	public int ldim; 
	
	/**
	 * Ctor from host data
	 */
	public FloatMat(float[] host, int row, int col)
	{
		this.host = host;
		initDim(row, col);
	}
	
	/**
	 * Ctor from 2D host data
	 */
	public FloatMat(float[][] host)
	{
		this(flatten(host), host.length, host[0].length);
	}
	
	/**
	 * Ctor for 1D vector (column vector)
	 */
	public FloatMat(float[] host)
	{
		this(host, host.length, 1);
	}

	/**
	 * Ctor from device data
	 */
	public FloatMat(Pointer device, int row, int col)
	{
		this.device = device;
		initDim(row, col);
	}
	
	/**
	 * Ctor that initializes device to 0
	 */
	public FloatMat(int row, int col)
	{
		this.device = GpuUtil.createDeviceFloat(row * col, true);
		initDim(row, col);
	}
	
	/**
	 * Copy ctor for clone()
	 */
	private FloatMat(FloatMat other)
	{
		this.host = other.host;
		this.device = other.device;
		this.row = other.row;
		this.col = other.col;
		this.ldim = other.ldim;
	}
	
	// Ctor helper
	private void initDim(int row, int col)
	{
		this.row = row;
		this.col = col;
		this.ldim = row;
	}
	
	/**
	 * Transpose the matrix and return a new one
	 * Nothing in the real data actually changes, but only a flag
	 * @return new instance
	 */
	public FloatMat transpose()
	{
		FloatMat mat = new FloatMat(this);
		mat.op = (op != CUBLAS_OP_N) ? 
				CUBLAS_OP_N : CUBLAS_OP_T;
		// Swap row and col dimension
		mat.row = this.col;
		mat.col = this.row;
		return mat;
	}
	
	public int getOp() {	return this.op;	}
	
	/**
	 * Invariant to transpose
	 */
	public int getOriginalRow()
	{
		return op == CUBLAS_OP_N ? row : col;
	}

	/**
	 * Invariant to transpose
	 */
	public int getOriginalCol()
	{
		return op == CUBLAS_OP_N ? col : row;
	}
	
	/**
	 * Get the device pointer
	 * If not currently on Cublas, we copy it to GPU
	 */
	public Pointer getDevice()
	{
		if (device == null)
			device = GpuBlas.toCublasFloat(host);
		return device;
	}

	/**
	 * Get the host pointer
	 * If not currently populated, we copy it to CPU
	 */
	public float[] getHost()
	{
		if (host == null)
		{
			host = new float[row * col];
			GpuBlas.toHostFloat(device, host);
		}
		return host;
	}
	
	/**
	 * Free the device pointer
	 */
	public void destroy()
	{
		host = null;
		cudaFree(device);
	}
	
	/**
	 * Utility: flatten a 2D float array to 1D, column major
	 */
	public static float[] flatten(float[][] A)
	{
		int row = A.length;
		int col = A[0].length;
		float[] ans = new float[row * col];
		int pt = 0;

		for (int j = 0; j < col; j ++)
			for (int i = 0; i < row; i ++)
				ans[pt ++] = A[i][j];

		return ans;
	}
	
	/**
	 * Utility: deflatten a 1D float to 2D matrix, column major
	 */
	public static float[][] deflatten(float[] A, int row)
	{
		int col = A.length / row;
		float[][] ans = new float[row][col];
		int pt = 0;
		
		for (int j = 0; j < col; j ++)
			for (int i = 0; i < row; i ++)
				ans[i][j] = A[pt ++];

		return ans;
	}
	
	/**
	 * Deflatten this to a 2D float array, column major
	 */
	public float[][] deflatten()
	{
		return deflatten(getHost(), this.row);
	}
}
