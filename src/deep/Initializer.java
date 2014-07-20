package deep;

import java.util.ArrayList;

import utils.CpuUtil;
import utils.MiscUtil;
import gpu.*;
import deep.units.ParamUnit;

public abstract class Initializer
{
	private boolean hasBias = true;
	
	/**
	 * All anonymous class needs to implement this
	 */
	public abstract void init(FloatMat w);
	
	/**
	 * @param hasBias if true, we'd assume W already has an extra vacant row
	 * and then set the last row to 0
	 * Then mult W * x will also have an extra row
	 */
	public void setBias(boolean hasBias) {	this.hasBias = hasBias;	}

	/**
	 * If hasBias is true, we'd assume W already has an extra vacant row
	 * and then set the last row to 0
	 * Then mult W * x will also have an extra row
	 */
	public void init(ParamUnit W)
	{	
		this.init(W.data());
		if (hasBias)
    		W.data().fillLastRow0();
	}
	
	
	private static final GpuRand gRand = new GpuRand(GpuRand.SEED);
	
	/**
	 * Does absolutely nothing
	 */
	public static final Initializer dummyIniter = 
			new Initializer() {
        		@Override
        		public void init(FloatMat w) { }
        	};
	
	/**
	 * Reset random generator to default seed
	 */
	public static void resetRand()
	{
		gRand.resetSeed(GpuRand.SEED);
	}

	/**
	 * Initialize uniform random parameter matrix from low to high
	 */
	public static Initializer uniformRandIniter(final float low, final float high)
	{
		final float range = high - low;
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				gRand.genUniformFloat(w);
				if (range != 1f || low != 0f)
					w.linear(range, low);
			}
		};
	}
	
	/**
	 * Initialize uniform random parameter matrix from -symmetric to +symmetric
	 */
	public static Initializer uniformRandIniter(final float symmetric)
	{
		return uniformRandIniter(-symmetric, symmetric);
	}
	
	/**
	 * Initialize (fill) with one same value
	 */
	public static Initializer fillIniter(final float val)
	{
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				w.fill(val);
			}
		};
	}
	
	
	// ******************** Distribution Initers ********************/
	/**
	 * Initialize with standard Gaussian distribution
	 */
	public static Initializer gaussianIniter(final float scalor)
	{
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				gRand.genNormalFloat(w);
				GpuBlas.scale(w, scalor);
			}
		};
	}
	
	/**
	 * Initialize with standard Laplacian distribution
	 */
	public static Initializer laplacianIniter(final float scalor)
	{
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				gRand.genLaplacianFloat(w);
				GpuBlas.scale(w, scalor);
			}
		};
	}
	
	/**
	 * Initialize with standard Cauchy distribution
	 */
	public static Initializer cauchyIniter(final float scalor)
	{
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				gRand.genCauchyFloat(w);
				GpuBlas.scale(w, scalor);
			}
		};
	}
	
	// ******************** Rahimi-Recht Fourier projection kernels ********************/
	public static enum ProjKernel { Gaussian, Laplacian, Cauchy }
	
	// Helper: used to set the last column of a distr-inited W to Uniform[0, 2*PI]
	// a pure distribution initer => transform this initer
	// Assumes an extra vacant row of 0 if hasBias is true
	private static Initializer projKernelAggregIniter(final Initializer distrIniter)
	{
		Initializer origIniter = new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				distrIniter.init(w);
				// Set last column to U[0, 2*PI] according to Rahimi-Recht
				gRand.genUniformFloat(
						w.createColOffset(-1), 0, 2 * Math.PI);
			}
		};
		return origIniter;
	}
	
	/**
	 * Kernel projectors will need gamma -> scalor multiplier for distrIniters
	 */
	public static float gammaToScalor(float gamma, ProjKernel kernelType)
	{
		switch (kernelType)
		{
		case Gaussian: return (float) Math.sqrt(2 * gamma);
		case Laplacian: return gamma;
		case Cauchy: return (float) Math.sqrt(gamma);
		default: return 0;
		}
	}
	
	/**
	 * Initialize Rahimi-Recht projection gaussian kernel, 
	 * which is just gaussian distr itself
	 * Assumes an extra vacant row of 0 if hasBias is true
	 * @param gamma scalor = sqrt(2 * gamma)
	 */
	public static Initializer gaussianProjKernelIniter(float gamma)
	{
		return projKernelAggregIniter(
				gaussianIniter(
						gammaToScalor(gamma, ProjKernel.Gaussian)));
	}
	
	/**
	 * Initialize Rahimi-Recht projection laplacian kernel, 
	 * which is just cauchy distr
	 * Assumes an extra vacant row of 0 if hasBias is true
	 * @param gamam scalor = gamma
	 */
	public static Initializer laplacianProjKernelIniter(float gamma)
	{
		return projKernelAggregIniter(
				cauchyIniter(
						gammaToScalor(gamma, ProjKernel.Laplacian)));
	}

	/**
	 * Initialize Rahimi-Recht projection cauchy kernel, 
	 * which is just laplacian distr
	 * Assumes an extra vacant row of 0 if hasBias is true
	 * @param gamam scalor = sqrt(gamma)
	 */
	public static Initializer cauchyProjKernelIniter(float gamma)
	{
		return projKernelAggregIniter(
				laplacianIniter(
						gammaToScalor(gamma, ProjKernel.Cauchy)));
	}
	
	/**
	 * Assumes an extra vacant row of 0 if hasBias is true
	 */
	public static Initializer projKernelIniter(ProjKernel type, float gamma)
	{
		switch (type)
		{
		case Gaussian : return gaussianProjKernelIniter(gamma); 
		case Laplacian : return laplacianProjKernelIniter(gamma); 
		case Cauchy : return cauchyProjKernelIniter(gamma); 
		}
		return null;
	}

	/**
	 * Aggregate Initer to mix different Rahimi-Recht kernels
	 * Assumes an extra vacant row of 0 if hasBias is true
	 * Sets the last column to be U[0, 2*PI]
	 * @param distrIniters: each pure-distr initer is responsible for initing a range of ROWs.
	 * can be repeated, e.g. lap for the first 30 rows, then gaussian 50 rows, then lap again 40 rows
	 * @param relativeRatios: divide (W.row - 1) into regions for which each initers are responsible
	 */
	public static Initializer mixProjKernelAggregIniter(
			final Initializer[] distrIniters, final double ... relativeRatios)
	{
		if (distrIniters.length != relativeRatios.length)
			throw new DeepException("Number of kernel initers must match number of relative ratios");
		
		// Normalize relativeRatios so that they sum up to 1
		double sum = 0;
		for (double d : relativeRatios)	sum += d;
		int i = 0;
		for (double d : relativeRatios)	relativeRatios[i++] = d / sum;
		
		Initializer mixOrigIniter = new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				// wt will soon be a deep transpose of W
				FloatMat wt = new FloatMat(w.transpose());
				// wt's col will be w's row. So filling wt's col region by region is the same as 
				// filling w's row region by region.
				int nIniters = distrIniters.length;
				int colCur = 0; // current column index
				final int colTotal = wt.col - 1; // reserve an extra row in 'w' for bias units
				for (int i = 0; i < nIniters; ++i)
				{
					if (i != nIniters -1) // not the last initer
					{
						int colRange = (int) Math.round(relativeRatios[i] * colTotal);
						distrIniters[i].init(wt.createColOffset(colCur, colCur + colRange));
						colCur += colRange;
					}
					else // the last initer inits any remaining uninitialized cols
						distrIniters[i].init(wt.createColOffset(colCur, colTotal));
				}
				// Give it back to w
				Thrust.transpose(wt, w);
				wt.destroy();
			}
		};
		
		// Post-process the initer to handle bias units and the extra U[0, 2*PI] col
		return projKernelAggregIniter(mixOrigIniter);
	}
	
	/**
	 * Use the pre-set enums to replace Initializer[]
	 * @param gammas for each kernel
	 * @see Initializer#mixProjKernelAggregIniter(ArrayList, ArrayList)
	 */
	public static Initializer mixProjKernelAggregIniter(
			ProjKernel[] projKernels, float[] gammas, double ... relativeRatios)
	{
		int N = projKernels.length;
		Initializer[] distrIniters = new Initializer[N];
		ProjKernel type;
		for (int i = 0; i < N; ++i)
		{
			type = projKernels[i];
			float scalor = gammaToScalor(gammas[i], type);
			switch (type)
			{
			case Gaussian : distrIniters[i] = gaussianIniter(scalor); break;
			case Laplacian : distrIniters[i] = cauchyIniter(scalor); break;
			case Cauchy: distrIniters[i] = laplacianIniter(scalor); break;
			}
		}
		return mixProjKernelAggregIniter(distrIniters, relativeRatios);
	}

	/**
	 * Use the pre-set enums to replace Initializer[]
	 * @param gamma common for all kernel
	 * @see Initializer#mixProjKernelAggregIniter(ArrayList, ArrayList)
	 */
	public static Initializer mixProjKernelAggregIniter(
			ProjKernel[] projKernels, float gamma, double ... relativeRatios)
	{
		return mixProjKernelAggregIniter(
				projKernels,  
				MiscUtil.repeatedArray(gamma, projKernels.length),
				relativeRatios);
	}
}
