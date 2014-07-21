package deep;

public abstract class LrScheme extends LearningPlan.Scheme
{
	/**
	 * Public main interface
	 */
	public final void updateBatch()
	{
		this.plan.lr = this.updateBatch_(this.plan);
	}

	/**
	 * Public main interface
	 */
	public final void updateEpoch()
	{
		this.plan.lr = this.updateEpoch_(this.plan);
	}
	
	/**
	 * Update learning rate between mini-batches
	 * @return new learning rate
	 */
	public abstract float updateBatch_(LearningPlan plan);

	/**
	 * Update learning rate over epochs 
	 * @return new learning rate
	 */
	public abstract float updateEpoch_(LearningPlan plan);
	
	/**
	 * @return default lr: doesn't change
	 */
	public float defaultLr() {	return plan.lr;	}
	
	/**
	 * @return Preset learning scheme: never decay, do nothing
	 */
	public static LrScheme dummyScheme()
	{
		return new LrScheme() {
			@Override
			public float updateEpoch_(LearningPlan plan)
			{ return this.defaultLr(); }
			@Override
			public float updateBatch_(LearningPlan plan)
			{ return this.defaultLr(); }
		};
	}
	
	/**
	 * @return Preset learning scheme: decay over every minibatch. 
	 */
	public static LrScheme constantDecayScheme()
	{
		return new LrScheme() {
			@Override
			public float updateEpoch_(LearningPlan plan)
			{
				return this.defaultLr();
			}
			@Override
			public float updateBatch_(LearningPlan plan)
			{
				float epochFraction = 
						(1f * plan.curEpoch * plan.totalSampleSize + plan.doneSampleSize) / plan.totalSampleSize;
				return plan.lrStart / (1 + epochFraction);
			}
		};
	}

	/**
	 * TODO
	 * @return Preset learning scheme: 
	 * let L2 = heldout loss of this epoch, and 
	 * let L1 = heldout loss of the last epoch
	 * In order to keep the same learning rate, we require that L2 >= L1(1+eps) 
	 * So, if L2 < L1(1+eps), we should decay the learning rate. 
	 * Equivalently, we decay if L2/L1 - 1 > -eps (because L1 is negative, change inequality direction) 
	 * if L2 < L1, then L2/L1 > 1 then L2/L1 - 1 > 0, which is unacceptable, perform decay!
	 */
	public static LrScheme epochDecayScheme(float improvementTol, float decayRate)
	{
		return new LrScheme() {
			@Override
			public float updateEpoch_(LearningPlan plan)
			{
				return 0;
			}
			@Override
			public float updateBatch_(LearningPlan plan)
			{
				return this.defaultLr();
			}
		};
	}
}
