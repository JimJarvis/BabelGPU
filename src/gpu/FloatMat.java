package gpu;

import utils.GpuUtil;
import utils.PP;
import jcuda.Pointer;
import static jcuda.runtime.JCuda.*;

/**
 * Struct around a matrix with row/col dimension info
 */
public class FloatMat
{
	private float[] host = null;
	private Pointer device = null;
	public int row;
	public int col;
	
	/**
	 * Ctor from host data
	 */
	public FloatMat(float[] host, int row, int col)
	{
		this.host = host;
		this.row = row;
		this.col = col;
	}

	/**
	 * Ctor from device data
	 */
	public FloatMat(Pointer device, int row, int col)
	{
		this.device = device;
		this.row = row;
		this.col = col;
	}
	
	/**
	 * Ctor that initializes device to 0
	 */
	public FloatMat(int row, int col)
	{
		this.device = GpuUtil.createDeviceFloat(row * col, true);
		this.row = row;
		this.col = col;
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
}
