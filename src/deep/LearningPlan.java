package deep;

import java.util.Iterator;

public class LearningPlan implements Iterable<Integer>
{
	/**
	 * Learning rate
	 */
	public float lr;
	/**
	 * Regularization
	 */
	public float reg;
	public int totalSampleSize;
	public int totalEpochs;
	/**
	 * number of samples already processed
	 */
	public int doneSampleSize = 0;
	public int curEpoch = 0;
	/**
	 * size of the batch just processed
	 * If set to negative or 0, infer curBatchSize from DataUnit.data.col
	 */
	public int curBatchSize = -1;
	
	public LearningPlan() {};
	
	public LearningPlan(float lr, float reg, int totalSampleSize, int totalEpochs)
	{
		this.lr = lr;
		this.reg = reg;
		this.totalSampleSize = totalSampleSize;
		this.totalEpochs = totalEpochs;
	}
	
	/**
	 * Copy ctor
	 * NOTE: variable states like curXXX aren't copied
	 */
	public LearningPlan(LearningPlan other)
	{
		this.lr = other.lr;
		this.reg = other.reg;
		this.totalSampleSize = other.totalSampleSize;
		this.totalEpochs = other.totalEpochs;
	}
	
	/**
	 * Prepare for re-run
	 */
	public void reset()
	{ 
		this.doneSampleSize = 0; 
	}
	
	/**
	 * How many samples left?
	 */
	public int remainTrainSize()
	{
		return this.totalSampleSize - this.doneSampleSize;
	}
	
	/**
	 * Do we use regularization?
	 */
	public boolean hasReg() { return this.reg > 0; }

	/**
	 * Iterates over each epoch
	 */
	@Override
	public Iterator<Integer> iterator()
	{
		return new Iterator<Integer>()
		{
			@Override
			public boolean hasNext()
			{
				return curEpoch < totalEpochs;
			}

			@Override
			public Integer next()
			{
				return curEpoch ++;
			}

			@Override
			public void remove() { }
		};
	}
	
	public String toString()
	{
		return String.format(
				"LearningPlan["
				+ "LR = %.3f\n"
				+ "Reg = %.3f\n"
				+ "TotalSampleSize = %d\n"
				+ "doneSampleSize = %d\n"
				+ "curBatchSize = %d\n" 
				+ "TotalEpochs = %d\n"
				+ "curEpoch = %d]",
				lr, reg, totalSampleSize, doneSampleSize, curBatchSize, totalEpochs, curEpoch);
	}
}