package utils;

import gpu.GpuException;
import jcuda.runtime.cudaError;
import jcuda.runtime.JCuda;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.cudaMemcpyKind;

public class GpuUtil
{
	/**
	 * cudaMemcpy from host pointer to device pointer
	 * @throws GpuException 
	 */
	public static void hostToDeviceFloat(
			Pointer devicePtr, Pointer hostPtr, int numFloatsToCopy) throws GpuException
	{
		GpuUtil.throwIfErrorCode(
			JCuda.cudaMemcpy(
				devicePtr, // dest 
				hostPtr,   // src
				numFloatsToCopy * Sizeof.FLOAT, 
				cudaMemcpyKind.cudaMemcpyHostToDevice
			)
		);
	}
	
	/**
	 * cudaMemcpy from host pointer to device pointer
	 * @throws GpuException 
	 */
	public static void deviceToHostFloat(
			Pointer devicePtr, Pointer hostPtr, int numFloatsToCopy) throws GpuException
	{
		GpuUtil.throwIfErrorCode(
			JCuda.cudaMemcpy(
				hostPtr,   // dest
				devicePtr, // src 
				numFloatsToCopy * Sizeof.FLOAT, 
				cudaMemcpyKind.cudaMemcpyDeviceToHost
			)
		);
	}
	
	/**
	 * cudaMemcpy from device pointer to host float array
	 * @throws GpuException 
	 */
	public static void deviceToHostFloat(
			Pointer devicePtr, /*ref*/ float[] hostArray) throws GpuException
	{
		GpuUtil.throwIfErrorCode(
			JCuda.cudaMemcpy(
				Pointer.to(hostArray), // dest 
				devicePtr, // src
        		hostArray.length * Sizeof.FLOAT, 
        		cudaMemcpyKind.cudaMemcpyDeviceToHost
        	)
        );
	}

	/**
	 * cudaMemcpy from device pointer to host int array
	 * @throws GpuException 
	 */
	public static void deviceToHostInt(
			Pointer devicePtr,  /*ref*/ int[] host) throws GpuException
	{
		GpuUtil.throwIfErrorCode(
			JCuda.cudaMemcpy(
				Pointer.to(host), // dest 
				devicePtr, // src
        		host.length * Sizeof.INT, 
        		cudaMemcpyKind.cudaMemcpyDeviceToHost
        	)
        );
	}
	
	/**
	 * cudaMemcpy from device pointer to host double array
	 * @throws GpuException 
	 */
	public static void deviceToHostDouble(
			Pointer devicePtr, /*ref*/ double[] host) throws GpuException
	{
		GpuUtil.throwIfErrorCode(
			JCuda.cudaMemcpy(
				Pointer.to(host), // dest 
				devicePtr, // src
        		host.length * Sizeof.DOUBLE, 
        		cudaMemcpyKind.cudaMemcpyDeviceToHost
			)
        );
	}
	
	/**
	 * A single float to a pointer wrapper on the host
	 */
	public static Pointer toFloatPointer(float a)
	{
		return Pointer.to(new float[] {a});
	}
	
	/**
	 * Create a float array on device
	 * @param memsetToZero true to initialize the memory to 0. Default false.
	 * @throws GpuException 
	 */
	public static Pointer allocateDeviceFloat(int n, boolean memsetToZero) throws GpuException
	{
		Pointer p = new Pointer();
		
		GpuUtil.throwIfErrorCode(
				JCuda.cudaMalloc(p, n * Sizeof.FLOAT)
			);
		
		if (memsetToZero)
		{
			GpuUtil.throwIfErrorCode(
					JCuda.cudaMemset(p, 0, n * Sizeof.FLOAT)
				);
		}
		return p;
	}

	/**
	 * Default: memset = false
	 * @throws GpuException 
	 */
	public static Pointer allocateDeviceFloat(int n) throws GpuException
	{
		return allocateDeviceFloat(n, false);
	}
	
	/**
	 * Create an int array on device
	 * @param memsetToZero true to initialize the memory to 0. Default false.
	 * @throws GpuException 
	 */
	public static Pointer allocateDeviceInt(int n, boolean memsetToZero) throws GpuException
	{
		Pointer p = new Pointer();
		
		GpuUtil.throwIfErrorCode(
				JCuda.cudaMalloc(p, n * Sizeof.INT)
			);
		
		if (memsetToZero)
		{
			GpuUtil.throwIfErrorCode(
					JCuda.cudaMemset(p, 0, n * Sizeof.INT)
				);
		}
		
		return p;
	}
	
	/**
	 * Clear the memory of a device pointer to 0
	 */
	public static void clearDeviceFloat(Pointer device, int size)
	{
		JCuda.cudaMemset(device, 0, size * Sizeof.FLOAT);
	}

	/**
	 * Default: memset = false
	 * @throws GpuException 
	 */
	public static Pointer allocateDeviceInt(int n) throws GpuException
	{
		return allocateDeviceInt(n, false);
	}
	
	private static void throwIfErrorCode(int exitStatus) throws GpuException 
	{
		if(exitStatus != cudaError.cudaSuccess)
		{
			throw new GpuException("Error code: " + exitStatus);
		}
	}
}
