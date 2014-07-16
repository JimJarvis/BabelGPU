package deep;

import java.util.Iterator;

public class LearningPlan implements Iterable<Integer>
{
	public float lr; // learning rate
	public float reg; // regularization
	public int totalTrainSize;
	public int totalEpochs;
	public int curTrainSize = 0;
	public int curEpoch = 0;
	
	public LearningPlan() {};
	
	public LearningPlan(float lr, float reg, int totalTrainSize, int totalEpochs)
	{
		this.lr = lr;
		this.reg = reg;
		this.totalTrainSize = totalTrainSize;
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
		this.totalTrainSize = other.totalTrainSize;
		this.totalEpochs = other.totalEpochs;
	}
	
	/**
	 * Prepare for re-run
	 */
	public void reset() { this.curTrainSize = 0; }
	
	/**
	 * Do we use regularization?
	 */
	public boolean hasReg()
	{
		return this.reg > 0;
	}

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
}
