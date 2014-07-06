package deep;

import gpu.*;
import deep.units.ParamUnit;

public abstract class Initializer
{
	public abstract void init(ParamUnit W);
	private static final long seed = 33760737L;
	
	private static final GpuRand rand = new GpuRand(seed);
	
	/**
	 * Does absolutely nothing
	 */
	public static final Initializer DUMMY = 
			new Initializer()
        	{
        		@Override
        		public void init(ParamUnit W) { }
        	};
	
	/**
	 * Initialize uniform random parameter matrix from low to high
	 */
	public static Initializer uniformRandInitializer(final float low, final float high)
	{
		final float range = high - low;
		return new Initializer()
		{
			@Override
			public void init(ParamUnit W)
			{
				FloatMat w = W.data;
				rand.genUniformFloat(w);
				if (range != 1f || low != 0f)
					w.linear(range, low);
			}
		};
	}
	
	/**
	 * Initialize uniform random parameter matrix from -symmetric to +symmetric
	 */
	public static Initializer uniformRandInitializer(final float symmetric)
	{
		return uniformRandInitializer(-symmetric, symmetric);
	}
	
	/**
	 * Reset random generator to default seed
	 */
	public static void resetRand()
	{
		rand.resetSeed(seed);
	}
}
