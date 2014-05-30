package gpu;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.Sizeof.*;

import java.util.Random;

import utils.GpuUtil;
import jcuda.Pointer;
import jcuda.jcurand.curandGenerator;

/**
 * Generates random floats on GPU
 */
public class GpuRand
{
	private curandGenerator generator;
	
	/**
	 * Ctor with seed
	 */
	public GpuRand(long seed)
	{
		generator = new curandGenerator();
		curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, seed);
	}
	
	/**
	 * Ctor with random seed
	 */
	private static Random rand = new Random();
	public GpuRand()
	{
		this(rand.nextLong());
	}

	/**
	 * Dtor
	 * Call after you're done with the random generator
	 */
	public void destroy()
	{
		if (generator != null)
    		curandDestroyGenerator(generator);
	}

	/**
	 * Generate a FloatMat with random uniform float
	 */
	public FloatMat genUniformFloat(int row, int col)
	{
		int n = row * col;
		Pointer device = GpuUtil.allocDeviceFloat(n);
		curandGenerateUniform(generator, device, n);
		return new FloatMat(device, row, col);
	}
	
	/**
	 * Generate a FloatMat (vector) with random uniform float
	 */
	public FloatMat genUniformFloat(int n)
	{
		return genUniformFloat(n, 1);
	}
	
	/**
	 * Generate a FloatMat with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genNormalFloat(int row, int col, float mean, float stdev)
	{
		int n = row * col;
		Pointer device = GpuUtil.allocDeviceFloat(n);
		curandGenerateNormal(generator, device, n, mean, stdev);
		return new FloatMat(device, row, col);
	}
	
	/**
	 * Generate a FloatMat (vector) with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genNormalFloat(int n, float mean, float stdev)
	{
		return genNormalFloat(n, 1, mean, stdev);
	}

	/**
	 * Generate a FloatMat with lognormal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genLogNormalFloat(int row, int col, float mean, float stdev)
	{
		int n = row * col;
		Pointer device = GpuUtil.allocDeviceFloat(n);
		curandGenerateLogNormal(generator, device, n, mean, stdev);
		return new FloatMat(device, row, col);
	}
	
	/**
	 * Generate a FloatMat (vector) with lognormal distribution 
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genLogNormalFloat(int n, float mean, float stdev)
	{
		return genLogNormalFloat(n, 1, mean, stdev);
	}
	
	/**
	 * Generate a FloatMat (int valued) with Poisson distribution
	 * @param lambda 
	 */
	public FloatMat genPoissonFloat(int row, int col, double lambda)
	{
		int n = row * col;
		Pointer device = GpuUtil.allocDeviceFloat(n);
		curandGeneratePoisson(generator, device, n, lambda);
		return new FloatMat(device, row, col);
	}
	
	/**
	 * Generate a FloatMat (vector, int valued) with Poisson distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genPoissonFloat(int n, double lambda)
	{
		return genPoissonFloat(n, 1, lambda);
	}

	
	//**************************************************/
	//******************* DOUBLE *******************/
	//**************************************************/
	/**
	 * Generate a DoubleMat with random uniform double
	 */
	public DoubleMat genUniformDouble(int row, int col)
	{
		int n = row * col;
		Pointer device = GpuUtil.allocDeviceDouble(n);
		curandGenerateUniformDouble(generator, device, n);
		return new DoubleMat(device, row, col);
	}
	
	/**
	 * Generate a DoubleMat (vector) with random uniform double
	 */
	public DoubleMat genUniformDouble(int n)
	{
		return genUniformDouble(n, 1);
	}
	
	/**
	 * Generate a DoubleMat with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genNormalDouble(int row, int col, double mean, double stdev)
	{
		int n = row * col;
		Pointer device = GpuUtil.allocDeviceDouble(n);
		curandGenerateNormalDouble(generator, device, n, mean, stdev);
		return new DoubleMat(device, row, col);
	}
	
	/**
	 * Generate a DoubleMat (vector) with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genNormalDouble(int n, double mean, double stdev)
	{
		return genNormalDouble(n, 1, mean, stdev);
	}

	/**
	 * Generate a DoubleMat with lognormal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genLogNormalDouble(int row, int col, double mean, double stdev)
	{
		int n = row * col;
		Pointer device = GpuUtil.allocDeviceDouble(n);
		curandGenerateLogNormalDouble(generator, device, n, mean, stdev);
		return new DoubleMat(device, row, col);
	}
	
	/**
	 * Generate a DoubleMat (vector) with lognormal distribution 
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genLogNormalDouble(int n, double mean, double stdev)
	{
		return genLogNormalDouble(n, 1, mean, stdev);
	}
}
