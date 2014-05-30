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
	private Pointer device;
	private float[] host;
	private int N; // size
	private boolean automaticFree = true;
	
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
	
	// helper
	private void init(int N)
	{
		this.N = N;
		host = null;
		if (automaticFree && device != null)
			cudaFree(device);
		device = new Pointer();
		cudaMalloc(device, N * FLOAT);
	}
	
	/**
	 * Generate n random uniform floats to device pointer
	 */
	public void genUniformFloat(int N)
	{
		init(N);
		curandGenerateUniform(generator, device, N);
		automaticFree = true;
	}
	
	/**
	 * Generate n normally distributed floats to device pointer
	 * @param mean 
	 * @param stdev standard deviation
	 */
	public void genNormalFloat(int N, float mean, float stdev)
	{
		init(N);
		curandGenerateNormal(generator, device, N, mean, stdev);
		automaticFree = true;
	}
	
	/**
	 * If you explicitly get the pointer,
	 * a flag will be set and you'd have to manually cudaFree() this pointer
	 */
	public Pointer getDevice()
	{
		automaticFree = false;
		return device;
	}
	
	/**
	 * After genFloat
	 */
	public float[] getHost()
	{
		if (host != null)
			return host;
		
		return host = GpuUtil.deviceToHostFloat(device, N);
	}
	
	/**
	 * Call after you're done with the random generator
	 */
	public void destroy()
	{
		if (generator != null)
    		curandDestroyGenerator(generator);
		if (automaticFree && device != null)
			cudaFree(device);
	}
}
