package gpu;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.*;

import java.util.Random;

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
		resetSeed(seed);
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
	 * Re-initialize with a specified seed
	 */
	public void resetSeed(long seed)
	{
		curandSetPseudoRandomGeneratorSeed(generator, seed);
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
	 * Fill a FloatMat with random uniform float
	 * @return input FloatMat A
	 */
	public FloatMat genUniformFloat(FloatMat A)
	{
		curandGenerateUniform(generator, A.toDevice(), A.size());
		return A;
	}

	/**
	 * Generate a new FloatMat with random uniform float
	 */
	public FloatMat genUniformFloat(int row, int col)
	{
		return genUniformFloat(new FloatMat(row, col, false));
	}
	
	/**
	 * Generate a new FloatMat (vector) with random uniform float
	 */
	public FloatMat genUniformFloat(int n)
	{
		return genUniformFloat(n, 1);
	}
	
	/**
	 * Fill a FloatMat with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genNormalFloat(FloatMat A, float mean, float stdev)
	{
		curandGenerateNormal(generator, A.toDevice(), A.size(), mean, stdev);
		return A;
	}
	
	/**
	 * Generate a new FloatMat with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genNormalFloat(int row, int col, float mean, float stdev)
	{
		return genNormalFloat(new FloatMat(row, col, false), mean, stdev);
	}
	
	/**
	 * Generate a new FloatMat (vector) with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genNormalFloat(int n, float mean, float stdev)
	{
		return genNormalFloat(n, 1, mean, stdev);
	}
	
	/**
	 * Fill a FloatMat with lognormal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genLogNormalFloat(FloatMat A, float mean, float stdev)
	{
		curandGenerateLogNormal(generator, A.toDevice(), A.size(), mean, stdev);
		return A;
	}

	/**
	 * Generate a new FloatMat with lognormal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genLogNormalFloat(int row, int col, float mean, float stdev)
	{
		return genLogNormalFloat(new FloatMat(row, col, false), mean, stdev);
	}
	
	/**
	 * Generate a new FloatMat (vector) with lognormal distribution 
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genLogNormalFloat(int n, float mean, float stdev)
	{
		return genLogNormalFloat(n, 1, mean, stdev);
	}
	
	/**
	 * Fill a FloatMat (int valued) with Poisson distribution
	 * @param lambda 
	 */
	public FloatMat genPoissonFloat(FloatMat A, double lambda)
	{
		curandGeneratePoisson(generator, A.toDevice(), A.size(), lambda);
		return A;
	}
	
	/**
	 * Generate a new FloatMat (int valued) with Poisson distribution
	 * @param lambda 
	 */
	public FloatMat genPoissonFloat(int row, int col, double lambda)
	{
		return genPoissonFloat(new FloatMat(row, col, false), lambda);
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
	 * Fill a DoubleMat with random uniform double
	 * @return input DoubleMat A
	 */
	public DoubleMat genUniformDouble(DoubleMat A)
	{
		curandGenerateUniformDouble(generator, A.toDevice(), A.size());
		return A;
	}

	/**
	 * Generate a new DoubleMat with random uniform double
	 */
	public DoubleMat genUniformDouble(int row, int col)
	{
		return genUniformDouble(new DoubleMat(row, col, false));
	}
	
	/**
	 * Generate a new DoubleMat (vector) with random uniform double
	 */
	public DoubleMat genUniformDouble(int n)
	{
		return genUniformDouble(n, 1);
	}
	
	/**
	 * Fill a DoubleMat with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genNormalDouble(DoubleMat A, double mean, double stdev)
	{
		curandGenerateNormalDouble(generator, A.toDevice(), A.size(), mean, stdev);
		return A;
	}
	
	/**
	 * Generate a new DoubleMat with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genNormalDouble(int row, int col, double mean, double stdev)
	{
		return genNormalDouble(new DoubleMat(row, col, false), mean, stdev);
	}
	
	/**
	 * Generate a new DoubleMat (vector) with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genNormalDouble(int n, double mean, double stdev)
	{
		return genNormalDouble(n, 1, mean, stdev);
	}
	
	/**
	 * Fill a DoubleMat with lognormal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genLogNormalDouble(DoubleMat A, double mean, double stdev)
	{
		curandGenerateLogNormalDouble(generator, A.toDevice(), A.size(), mean, stdev);
		return A;
	}

	/**
	 * Generate a new DoubleMat with lognormal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genLogNormalDouble(int row, int col, double mean, double stdev)
	{
		return genLogNormalDouble(new DoubleMat(row, col, false), mean, stdev);
	}
	
	/**
	 * Generate a new DoubleMat (vector) with lognormal distribution 
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public DoubleMat genLogNormalDouble(int n, double mean, double stdev)
	{
		return genLogNormalDouble(n, 1, mean, stdev);
	}
}
