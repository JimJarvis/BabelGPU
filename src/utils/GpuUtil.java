package utils;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.Sizeof.*;
import static utils.CpuUtil.matAvgDiff;
import gpu.*;
import jcuda.*;
import jcuda.jcublas.*;
import jcuda.jcurand.JCurand;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

public class GpuUtil
{
	//**************************************************/
	//******************* FLOAT *******************/
	//**************************************************/
	/**
	 * cudaMemcpy from device pointer to host float array
	 */
	public static float[] deviceToHostFloat(Pointer device, int size)
	{
		 // Copy device memory to host 
		float[] host = new float[size];
        cudaMemcpy(Pointer.to(host), device, 
            size * FLOAT, cudaMemcpyDeviceToHost);
        
        return host;
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
	 */
	public static Pointer allocDeviceFloat(int size, boolean memsetToZero)
	{
		Pointer p = new Pointer();
		cudaMalloc(p, size * FLOAT);
		if (memsetToZero)
    		cudaMemset(p, 0, size * FLOAT);
		return p;
	}

	/**
	 * Default: memset = false
	 */
	public static Pointer allocDeviceFloat(int size)
	{
		return allocDeviceFloat(size, false);
	}
	
	/**
	 * Clear the memory of a device pointer to 0
	 */
	public static void clearDeviceFloat(Pointer device, int size)
	{
		cudaMemset(device, 0, size * FLOAT);
	}

	/**
	 * Check the gold standard generated from Matlab
	 * Assume the goldFile has extension ".txt"
	 * @param testName
	 * @param tol tolerance of error
	 */
	public static void checkGold(FloatMat gpu, String goldFile, String testName, float tol)
	{
		CsvReader csv = new CsvReader(goldFile + ".txt");
		float[][] Gold = csv.readFloatMat();
		float[][] Host = gpu.deflatten();
		
		float diff = matAvgDiff(Gold, Host);
		PP.setPrecision(3);
		PP.setScientific(true);
		
		PP.p("["+testName+"]", diff < tol ? "PASS:" : "FAIL:", diff);
	}

	//**************************************************/
	//******************* DOUBLE *******************/
	//**************************************************/
	/**
	 * cudaMemcpy from device pointer to host double array
	 */
	public static double[] deviceToHostDouble(Pointer device, int size)
	{
		 // Copy device memory to host 
		double[] host = new double[size];
        cudaMemcpy(Pointer.to(host), device, 
            size * DOUBLE, cudaMemcpyDeviceToHost);
        
        return host;
	}

	/**
	 * A single double to a pointer wrapper on the host
	 */
	public static Pointer toDoublePointer(double a)
	{
		return Pointer.to(new double[] {a});
	}
	
	/**
	 * Create a double array on device
	 * @param memsetToZero true to initialize the memory to 0. Default false.
	 */
	public static Pointer allocDeviceDouble(int size, boolean memsetToZero)
	{
		Pointer p = new Pointer();
		cudaMalloc(p, size * DOUBLE);
		if (memsetToZero)
    		cudaMemset(p, 0, size * DOUBLE);
		return p;
	}

	/**
	 * Default: memset = false
	 */
	public static Pointer allocDeviceDouble(int size)
	{
		return allocDeviceDouble(size, false);
	}

	/**
	 * Clear the memory of a device pointer to 0
	 */
	public static void clearDeviceDouble(Pointer device, int size)
	{
		cudaMemset(device, 0, size * DOUBLE);
	}
	
	/**
	 * Check the gold standard generated from Matlab
	 * Assume the goldFile has extension ".txt"
	 * @param testName
	 * @param tol tolerance of error
	 */
	public static void checkGold(DoubleMat gpu, String goldFile, String testName, double tol)
	{
		CsvReader csv = new CsvReader(goldFile + ".txt");
		double[][] Gold = csv.readDoubleMat();
		double[][] Host = gpu.deflatten();
		
		double diff = matAvgDiff(Gold, Host);
		PP.setPrecision(3);
		PP.setScientific(true);
		
		PP.p("["+testName+"]", diff < tol ? "PASS:" : "FAIL:", diff);
	}
	
	
	//**************************************************/
	//******************* COMMON *******************/
	//**************************************************/
	/**
	 * Wait until all threads are finished
	 */
	public static void synchronize()
	{
		cudaDeviceSynchronize();
	}
	
	/**
	 * Enable detailed exceptions
	 */
	public static void enableExceptions()
	{
		JCuda.setExceptionsEnabled(true);
		JCurand.setExceptionsEnabled(true);
		JCublas.setExceptionsEnabled(true);
		JCublas2.setExceptionsEnabled(true);
	}
	
	/**
	 * Device info struct
	 */
	public static class GpuInfo
	{
		public String DeviceName;
		public int[] Version; // [major_version, minor_version]

		// All memory will be in Kb
		public double ClockRate; // GHz
		public int RegisterPerBlock;
		public int RegisterPerMultiProcessor;
		public int ConcurrentKernels;
		public int MultiProcessorCount;
		public boolean Integrated;
		
		public int[] MaxBlockDim;
		public int[] MaxGridDim;
		public int MaxThreadsPerBlock;
		public int MaxThreadsPerMultiProcessor;

		public long GlobalMemory;
		public long SharedMemoryPerBlock; 
		public long SharedMemoryPerMultiprocessor;
		public long ConstantMemory;
		public long L2CacheSize;
		public double MemoryClockRate; // GHz
		public int MemoryBusWidth;
		public boolean ManagedMemory;
		public int WarpSize;

		
		public static int toMb(long kb) {	return (int) (kb/1024);	}
		public static int toGb(long kb) {	return (int) (kb/(1024 * 1024));	}
		
		public String toString()
		{
			PP.setPrecision(3);
			return PP.all2str(
					"Device name:", DeviceName, 
					"\nVersion:", Version[0]+"."+Version[1],
					"\nClock rate:", ClockRate, "GHz",
					"\nRegister per block:", RegisterPerBlock,
					"\nRegister per multiprocessor:", RegisterPerMultiProcessor,
					"\nConcurrent kernels:", ConcurrentKernels,
					"\nMultiprocessor count:", MultiProcessorCount,
					"\nIntegrated:", Integrated,
					"\nMax block dimension:", MaxBlockDim,
					"\nMax grid dimension:", MaxGridDim,
					"\nMax threads per block:", MaxThreadsPerBlock,
					"\nMax threads per multiprocessor:", MaxThreadsPerMultiProcessor,
					"\nGlobal memory:", toGb(GlobalMemory), "Gb",
					"\nShared memory per block", SharedMemoryPerBlock, "Kb",
					"\nShared memory per multiprocessor:", SharedMemoryPerMultiprocessor, "Kb",
					"\nConstant memory:", ConstantMemory, "Kb",
					"\nL2 cache size:", L2CacheSize, "Kb",
					"\nMemory clock rate:", MemoryClockRate, "GHz",
					"\nMemory bus width:", MemoryBusWidth,
					"\nManaged memory:", ManagedMemory,
					"\nWarp size:", WarpSize
					);
		}
	}
	
	/**
	 * @return GpuInfo struct for each GPU device
	 */
	public static GpuInfo[] getGpuInfo()
	{
		int deviceCount[] = new int[1];
		cudaGetDeviceCount(deviceCount);
		GpuInfo gpuInfos[] = new GpuInfo[deviceCount[0]];
		
		final int KB = 1024;
		final double GHz = 1e6;
		for (int i = 0; i < deviceCount[0]; i++)
		{
			// All memory will be in Kb
			GpuInfo gpuInfo = new GpuInfo();
			cudaDeviceProp prop = new cudaDeviceProp();
			cudaGetDeviceProperties(prop, i);
			gpuInfo.DeviceName = prop.getName();
			gpuInfo.Version = new int[] {prop.major, prop.minor};
			
			gpuInfo.ClockRate = prop.clockRate / GHz;
			gpuInfo.RegisterPerBlock = prop.regsPerBlock;
			gpuInfo.RegisterPerMultiProcessor = prop.regsPerMultiprocessor;
			gpuInfo.ConcurrentKernels = prop.concurrentKernels;
			gpuInfo.MultiProcessorCount = prop.multiProcessorCount;
			gpuInfo.Integrated = prop.integrated > 0;
			
			gpuInfo.MaxBlockDim = prop.maxThreadsDim;
			gpuInfo.MaxGridDim = prop.maxGridSize;
			gpuInfo.MaxThreadsPerBlock = prop.maxThreadsPerBlock;
			gpuInfo.MaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;

			gpuInfo.GlobalMemory = prop.totalGlobalMem / KB;
			gpuInfo.SharedMemoryPerBlock = prop.sharedMemPerBlock / KB;
			gpuInfo.SharedMemoryPerMultiprocessor = prop.sharedMemPerMultiprocessor / KB;
			gpuInfo.ConstantMemory = prop.totalConstMem / KB;
			gpuInfo.L2CacheSize = prop.l2CacheSize / KB;
			gpuInfo.MemoryClockRate = prop.memoryClockRate / GHz;
			gpuInfo.MemoryBusWidth = prop.memoryBusWidth;
			gpuInfo.ManagedMemory = prop.managedMemory > 0;
			gpuInfo.WarpSize = prop.warpSize;
			
			gpuInfos[i] = gpuInfo;
		}
		return gpuInfos;
	}
	
	
}
