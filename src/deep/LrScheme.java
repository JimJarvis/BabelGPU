package deep;

import java.util.ArrayList;

public abstract class LrScheme extends LearningPlan.Scheme
{
	private static final long serialVersionUID = 1L;

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
			private static final long serialVersionUID = 1L;
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
			private static final long serialVersionUID = 1L;
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
	 * let curL = heldout loss of this epoch, and 
	 * let lastL = heldout loss of the last epoch. 
	 * In order to keep the same learning rate, we require that curL < lastL *(1 - improvementTol).
	 * if doesn't hold, we decay lr *= decayRate 
	 * if curL > lastL (we're doing worse), roll back to last parameters, decay LR and try again
	 */
	public static LrScheme epochDecayScheme(final float improvementTol, final float decayRate)
	{
		return new LrScheme() {
			private static final long serialVersionUID = 1L;

			@Override
			public float updateEpoch_(LearningPlan plan)
			{
				DeepNet net = plan.net;
				// Loss-based dynamic decay cannot go without loss calculation
				if (! net.doesCalcLoss())
					return defaultLr();

				ArrayList<Float> record = plan.record;
				float curLoss = record.get(record.size()-1);
				float lastLoss = plan.curEpoch == 0 ?
						Float.POSITIVE_INFINITY :
						record.get(record.size() - 2);

				// If curLoss = INF, we're screwed
				// If lastLoss = INF, curLoss isn't INF, we are good and don't 
				boolean decay = Float.isInfinite(curLoss) 
						|| ! Float.isInfinite(lastLoss)
						&& (lastLoss - curLoss) / lastLoss < improvementTol;

				// Instead of making progress, we're actually doing worse
				boolean regress = plan.curEpoch != 0
						&& (Float.isInfinite(curLoss)
							|| curLoss > lastLoss);

				if (decay)
				{   // Roll-back params to last epoch if our progress is actually regressive
					if (regress)
						net.restoreLastEpochParams();
					else
						net.recordLastEpochParams();
					return plan.lr * decayRate;
				}
				else // no decay
				{ // Record lastest theta
					net.recordLastEpochParams();
					return defaultLr();
				}
			}
			
			@Override
			public float updateBatch_(LearningPlan plan)
			{
				return this.defaultLr();
			}
		};
	}
}
