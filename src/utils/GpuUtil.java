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

public class GpuUtil
{
	//******************* COMMON *******************/
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
	 * Show device details
	 */
	public static void gpuInfo()
	{
		//TODO
	}
	
	
	//**************************************************/
	//******************* FLOAT *******************/
	//**************************************************/
	/**
	 * cudaMemcpy from device pointer to host float array
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
	public static Pointer allocDeviceFloat(int n, boolean memsetToZero)
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
	public static Pointer allocDeviceFloat(int n)
	{
		return allocDeviceFloat(n, false);
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
	public static double[] deviceToHostDouble(Pointer device, int n)
	{
		 // Copy device memory to host 
		double[] host = new double[n];
        cudaMemcpy(Pointer.to(host), device, 
            n * DOUBLE, cudaMemcpyDeviceToHost);
        
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
	public static Pointer allocDeviceDouble(int n, boolean memsetToZero)
	{
		Pointer p = new Pointer();
		cudaMalloc(p, n * DOUBLE);
		if (memsetToZero)
    		cudaMemset(p, 0, n * DOUBLE);
		return p;
	}

	/**
	 * Default: memset = false
	 */
	public static Pointer allocDeviceDouble(int n)
	{
		return allocDeviceDouble(n, false);
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
}
