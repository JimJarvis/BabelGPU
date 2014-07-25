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
	/**
	 * Re-initialization seed
	 */
	public static final long SEED = 33760737L;
	private curandGenerator generator;
	private static Random rand;
	
	/**
	 * Ctor with seed
	 */
	public GpuRand(long seed)
	{
		rand = new Random(seed);
		createGenerator(seed);
	}
	
	private void createGenerator(long seed)
	{
		generator = new curandGenerator();
		curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, seed);
	}
	
	/**
	 * Ctor with random seed
	 */
	public GpuRand()
	{
		this(System.currentTimeMillis());
	}
	
	/**
	 * Re-initialize with a specified seed
	 * We actually destroy and reallocate the generator 
	 * because resetting seed doesn't ensure the same random sequence
	 */
	public void resetSeed(long seed)
	{
		destroy();
		createGenerator(seed);
		rand = new Random(seed);
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
	public FloatMat genUniformFloat(FloatMat A, double low, double high)
	{
		curandGenerateUniform(generator, A.toDevice(), A.size());
		A.linear((float)(high - low), (float)low);
		return A;
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
	 * Generate a new FloatMat with random uniform float
	 */
	public FloatMat genUniformFloat(int row, int col, double low, double high)
	{
		return genUniformFloat(new FloatMat(row, col, false), low, high);
	}
	
	/**
	 * Generate a new FloatMat (vector) with random uniform float
	 */
	public FloatMat genUniformFloat(int n)
	{
		return genUniformFloat(n, 1);
	}
	
	/**
	 * Thrust version: fill a FloatMat with normal distribution
	 * If too many elements (2^25), will take too long and trigger a CudaTimeout error. 
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genNormalFloat(FloatMat A, double mean, double stddev)
	{
		Thrust.fill_rand_normal(A, (float) mean, (float) stddev);
		return A;
	}
	
	/**
	 * cuRAND version: fill a FloatMat with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 * @deprecated doesn't work with createOffset 
	 * if the offset is not an even number, the method throws MisalignedAddress cuda exception 
	 * but faster than {@link #genUniformFloat(FloatMat) thrust version}
	 */
	public FloatMat genNormalFloatCurand(FloatMat A, double mean, double stdev)
	{
		// Ugliest hack in the history of programming
		// WARNING: curandGenNormal doesn't work on array with odd number of elements!!
		// Force even, then add 1 more at last
		int size = A.size();
		boolean odd = size % 2 == 1;
		curandGenerateNormal(generator, A.toDevice(), odd ? size - 1 : size, (float)mean, (float)stdev);
		if (odd) // manually set the last unfilled slot with rand normal
			A.setSingle(-1, (float) (rand.nextGaussian() * stdev + mean));
		return A;
	}
	
	/**
	 * Fill a FloatMat with standard normal distribution
	 */
	public FloatMat genNormalFloat(FloatMat A)
	{
		return genNormalFloat(A, 0, 1);
	}
	
	/**
	 * Generate a new FloatMat with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genNormalFloat(int row, int col, double mean, double stdev)
	{
		return genNormalFloat(new FloatMat(row, col, false), mean, stdev);
	}
	
	/**
	 * Generate a new FloatMat with standard normal distribution
	 */
	public FloatMat genNormalFloat(int row, int col)
	{
		return genNormalFloat(row, col, 0, 1);
	}
	
	/**
	 * Generate a new FloatMat (vector) with normal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genNormalFloat(int n, double mean, double stdev)
	{
		return genNormalFloat(n, 1, mean, stdev);
	}
	
	/**
	 * Generate a new FloatMat (vector) with standard normal distribution
	 */
	public FloatMat genNormalFloat(int n)
	{
		return genNormalFloat(n, 1, 0f, 1f);
	}
	
	/**
	 * Fill a FloatMat with lognormal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genLogNormalFloat(FloatMat A, double mean, double stdev)
	{
		// Ugliest hack in the history of programming
		// WARNING: curandGenLogNormal doesn't work on array with odd number of elements!!
		// Force even, then add 1 more at last
		int size = A.size();
		boolean odd = size % 2 == 1;
		curandGenerateLogNormal(generator, A.toDevice(), odd ? size - 1 : size, (float)mean, (float)stdev);
		if (odd) // manually set the last unfilled slot with rand lognormal
			A.setSingle(-1, (float) Math.exp(rand.nextGaussian() * stdev + mean));
		return A;
	}
	
	/**
	 * Fill a FloatMat with standard lognormal distribution
	 */
	public FloatMat genLogNormalFloat(FloatMat A)
	{
		return genLogNormalFloat(A, 0, 1);
	}

	/**
	 * Generate a new FloatMat with lognormal distribution
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genLogNormalFloat(int row, int col, double mean, double stdev)
	{
		return genLogNormalFloat(new FloatMat(row, col, false), mean, stdev);
	}
	
	/**
	 * Generate a new FloatMat with std lognormal distribution
	 */
	public FloatMat genLogNormalFloat(int row, int col)
	{
		return genLogNormalFloat(row, col, 0, 1);
	}
	
	/**
	 * Generate a new FloatMat (vector) with lognormal distribution 
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public FloatMat genLogNormalFloat(int n, double mean, double stdev)
	{
		return genLogNormalFloat(n, 1, mean, stdev);
	}
	
	/**
	 * Generate a new FloatMat (vector) with std lognormal distribution 
	 */
	public FloatMat genLogNormalFloat(int n)
	{
		return genLogNormalFloat(n, 1, 0f, 1f);
	}
	
	/**
	 * Generate a FloatMat with standard Laplacian distribution
	 */
	public FloatMat genLaplacianFloat(FloatMat A)
	{
		genUniformFloat(A);
		Thrust.laplacian(A);
		return A;
	}

	/**
	 * Generate a FloatMat with standard Laplacian distribution
	 * @return new
	 */
	public FloatMat genLaplacianFloat(int row, int col)
	{
		return genLaplacianFloat(new FloatMat(row, col, false));
	}
	
	/**
	 * Generate a FloatMat with standard Cauchy distribution
	 */
	public FloatMat genCauchyFloat(FloatMat A)
	{
		genUniformFloat(A);
		Thrust.cauchy(A);
		return A;
	}

	/**
	 * Generate a FloatMat with standard Cauchy distribution
	 * @return new
	 */
	public FloatMat genCauchyFloat(int row, int col)
	{
		return genCauchyFloat(new FloatMat(row, col, false));
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
