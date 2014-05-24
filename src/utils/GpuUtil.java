package utils;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.Sizeof.*;
import jcuda.Pointer;

public class GpuUtil
{
	/**
	 * From device pointer to host float array
	 */
	public static float[] deviceToHostFloat(Pointer device, int n)
	{
		 // Copy device memory to host 
		float[] host = new float[n];
        cudaMemcpy(Pointer.to(host), device, 
            n * FLOAT, cudaMemcpyDeviceToHost);
        
        return host;
	}

	/**
	 * From device pointer to host int array
	 */
	public static int[] deviceToHostInt(Pointer device, int n)
	{
		// Copy device memory to host 
		int[] host = new int[n];
		cudaMemcpy(Pointer.to(host), device, 
				n * INT, cudaMemcpyDeviceToHost);

		return host;
	}
	
	/**
	 * From device pointer to host double array
	 */
	public static double[] deviceToHostDouble(Pointer device, int n)
	{
		 // Copy device memory to host 
		double[] host = new double[n];
        cudaMemcpy(Pointer.to(host), device, 
            n * DOUBLE, cudaMemcpyDeviceToHost);
        
        return host;
	}
	
	/**
	 * A single float to host pointer
	 */
	public static Pointer toHostFloatPointer(float a)
	{
		return Pointer.to(new float[] {a});
	}
	
	/**
	 * Create a float array on device
	 * @param memsetToZero true to initialize the memory to 0. Default false.
	 */
	public static Pointer createDeviceFloat(int n, boolean memsetToZero)
	{
		Pointer p = new Pointer();
		cudaMalloc(p, n * FLOAT);
		if (memsetToZero)
    		cudaMemset(p, 0, n * FLOAT);
		return p;
	}

	/**
	 * Default: memset = false
	 */
	public static Pointer createDeviceFloat(int n)
	{
		return createDeviceFloat(n, false);
	}
	
	/**
	 * Create an int array on device
	 * @param memsetToZero true to initialize the memory to 0. Default false.
	 */
	public static Pointer createDeviceInt(int n)
	{
		Pointer p = new Pointer();
		cudaMalloc(p, n * INT);
		cudaMemset(p, 0, n * INT);
		return p;
	}
}
