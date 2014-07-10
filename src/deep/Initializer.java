package deep;

import java.util.ArrayList;

import utils.PP;
import gpu.*;
import deep.units.ParamUnit;

public abstract class Initializer
{
	public abstract void init(FloatMat w);
	
	public void init(ParamUnit W) {	this.init(W.data);  }
	
	
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
	
	/**
	 * Aggregate Initer to handle 'bias' units
	 * Assume W already has an extra vacant row
	 * Set the last row of an initiated parameter to be [0, 0, 0, ..., 1]
	 * The last element of the last row is 1
	 * In this way, multiplication W * x would preserve the last row of x, which should be all 1
	 */
	public static Initializer multBiasAggregIniter(final Initializer origIniter)
	{
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				origIniter.init(w);
				w.fillRow(0, -1);
				w.setSingle(-1, -1, 1);
			}
		};
	}
	
	// ******************** Distribution Initers ********************/
	/**
	 * Initialize with standard Gaussian distribution
	 * @param gamma scalor = sqrt(2 * gamma)
	 */
	public static Initializer gaussianIniter(double gamma)
	{
		final float scalor = (float) Math.sqrt(2 * gamma);
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				gRand.genNormalFloat(w);
				w.linear(scalor, 0);
			}
		};
	}
	
	/**
	 * Initialize with standard Laplacian distribution
	 * @param gamam scalor = gamma
	 */
	public static Initializer laplacianIniter(final double gamma)
	{
		final float scalor = (float) gamma;
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				gRand.genLaplacianFloat(w);
				w.linear(scalor, 0);
			}
		};
	}
	
	/**
	 * Initialize with standard Cauchy distribution
	 * @param gamam scalor = gamma
	 */
	public static Initializer cauchyIniter(double gamma)
	{
		final float scalor = (float) Math.sqrt(gamma);
		return new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				gRand.genCauchyFloat(w);
				w.linear(scalor, 0);
			}
		};
	}
	
	// ******************** Rahimi-Recht Fourier projection kernels ********************/
	public static enum ProjKernel { Gaussian, Laplacian, Cauchy }
	
	// Helper: used to set the last column of a distr-inited W to Uniform[0, 2*PI]
	// a pure distribution initer => transform this initer
	// Assumes an extra row of bias
	private static Initializer projKernelAggregIniter(final Initializer distrIniter)
	{
		Initializer origIniter = new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				distrIniter.init(w);
				// Set last column to U[0, 2*PI] according to Rahimi-Recht
				gRand.genUniformFloat(
						w.createColOffset(w.col-1, w.col), 0, 2 * Math.PI);
			}
		};
		return multBiasAggregIniter(origIniter);
	}
	
	/**
	 * Initialize Rahimi-Recht projection gaussian kernel, 
	 * which is just gaussian distr itself
	 * Assumes extra row of bias
	 */
	public static Initializer gaussianProjKernelIniter(final double gamma)
	{
		return projKernelAggregIniter(gaussianIniter(gamma));
	}
	
	/**
	 * Initialize Rahimi-Recht projection laplacian kernel, 
	 * which is just cauchy distr
	 * Assumes extra row of bias
	 */
	public static Initializer laplacianProjKernelIniter(final double gamma)
	{
		return projKernelAggregIniter(cauchyIniter(gamma));
	}

	/**
	 * Initialize Rahimi-Recht projection cauchy kernel, 
	 * which is just laplacian distr
	 * Assumes extra row of bias
	 */
	public static Initializer cauchyProjKernelIniter(final double gamma)
	{
		return projKernelAggregIniter(laplacianIniter(gamma));
	}
	
	/**
	 * Assumes extra row of bias
	 */
	public static Initializer projKernelIniter(ProjKernel type, final double gamma)
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
	 * Assume W has an extra row for handling bias units, calls multBiasAggregIniter
	 * Sets the last column to be U[0, 2*PI]
	 * @param distrIniters: each pure-distr initer is responsible for initing a range of ROWs.
	 * can be repeated, e.g. lap for the first 30 rows, then gaussian 50 rows, then lap again 40 rows
	 * @param relativeRatios: divide (W.row - 1) into regions for which each initers are responsible
	 */
	public static Initializer mixProjKernelAggregIniter(
			final ArrayList<Initializer> distrIniters, final ArrayList<Double> relativeRatios)
	{
		if (distrIniters.size() != relativeRatios.size())
			throw new DeepException("Number of kernel initers must match number of relative ratios");
		
		// Normalize relativeRatios so that they sum up to 1
		double sum = 0;
		for (double d : relativeRatios)	sum += d;
		int i = 0;
		for (double d : relativeRatios)	relativeRatios.set(i++, d / sum);
		
		Initializer mixOrigIniter = new Initializer() {
			@Override
			public void init(FloatMat w)
			{
				// wt will soon be a deep transpose of W
				FloatMat wt = new FloatMat(w.transpose());
				// wt's col will be w's row. So filling wt's col region by region is the same as 
				// filling w's row region by region.
				int nIniters = distrIniters.size();
				int colCur = 0; // current column index
				final int colTotal = wt.col - 1; // reserve an extra row in 'w' for bias units
				for (int i = 0; i < nIniters; ++i)
				{
					if (i != nIniters -1) // not the last initer
					{
						int colRange = (int) Math.round(relativeRatios.get(i) * colTotal);
						distrIniters.get(i).init(wt.createColOffset(colCur, colCur + colRange));
						colCur += colRange;
					}
					else // the last initer inits any remaining uninitialized cols
						distrIniters.get(i).init(wt.createColOffset(colCur, colTotal));
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
	 * Use the pre-set enums to replace ArrayList<Initializers>
	 * @param gamma assume common for all distrIniters
	 * @see Initializer#mixProjKernelAggregIniter(ArrayList, ArrayList)
	 */
	public static Initializer mixProjKernelAggregIniter(
			final ArrayList<ProjKernel> projKernels, double gamma, final ArrayList<Double> relativeRatios)
	{
		ArrayList<Initializer> distrIniters = new ArrayList<>();
		for (ProjKernel type : projKernels)
			switch (type)
			{
			case Gaussian : distrIniters.add(gaussianIniter(gamma)); break;
			case Laplacian : distrIniters.add(cauchyIniter(gamma)); break;
			case Cauchy: distrIniters.add(laplacianIniter(gamma)); break;
			}
		return mixProjKernelAggregIniter(distrIniters, relativeRatios);
	}
}
