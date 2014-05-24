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
	public static float[] toHostFloat(Pointer deviceData, int n)
	{
		 // Copy device memory to host 
		float[] hostData = new float[n];
        cudaMemcpy(Pointer.to(hostData), deviceData, 
            n * FLOAT, cudaMemcpyDeviceToHost);
        
        return hostData;
	}
	
	/**
	 * From device pointer to host double array
	 */
	public static double[] toHostDouble(Pointer deviceData, int n)
	{
		 // Copy device memory to host 
		double[] hostData = new double[n];
        cudaMemcpy(Pointer.to(hostData), deviceData, 
            n * DOUBLE, cudaMemcpyDeviceToHost);
        
        return hostData;
	}
	
	/**
	 * A single float to GPU pointer
	 */
	public static Pointer toPointer(float a)
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
}
