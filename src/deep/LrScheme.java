package deep;

public abstract class LrScheme
{
	/**
	 * Update learning rate between mini-batches
	 * @return new learning rate
	 */
	public abstract float updateBatch(LearningPlan plan);

	/**
	 * Update learning rate over epochs 
	 * @return new learning rate
	 */
	public abstract float updateEpoch(LearningPlan plan);
	
	/**
	 * Preset learning scheme
	 */
	public static final LrScheme ConstantDecayScheme = 
			new LrScheme()
			{
				@Override
				public float updateEpoch(LearningPlan plan)
				{
					return 0;
				}
				
				@Override
				public float updateBatch(LearningPlan plan)
				{
					return 0;
				}
			};

	/**
	 * Preset learning scheme
	 */
	public static final LrScheme EpochDecayScheme = 
			new LrScheme()
			{
				
				@Override
				public float updateEpoch(LearningPlan plan)
				{
					return 0;
				}
				
				@Override
				public float updateBatch(LearningPlan plan)
				{
					return 0;
				}
			};
}
