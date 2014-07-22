package deep;

import gpu.*;
import deep.units.*;

/**
 * Regularization scheme
 */
public abstract class RegScheme extends LearningPlan.Scheme
{
	private static final long serialVersionUID = 1L;
	/**
	 * Main interface to outside
	 */
	public final float regLoss()
	{
		DeepNet net = plan.net;
		return net.doesCalcLoss() && plan.hasReg() ?
				regLoss_(plan, net.getParams()) : 0;
	}
	
	/**
	 * Use it inside {@link ParamComputeUnit#backward()}
	 */
	public final void regGradUpdate(ParamComputeUnit pcUnit)
	{
		if (plan.hasReg())
    		regGradUpdate_(plan, pcUnit.W);
	}
	
	/**
	 * @return regularization loss
	 */
	protected abstract float regLoss_(LearningPlan plan, ParamList paramList);
	
	/**
	 * Regularize parameter gradient. Update the reg term. W += -lr * reg * W 
	 */
	protected abstract void regGradUpdate_(LearningPlan plan, ParamUnit W);
	
	/**
	 * Preset scheme: L-2 square-sum regularizer
	 * We make this a subclass because units like LinearUnit 
	 * can use the class info for further GPU optimization
	 */
	public static class L2RegScheme extends RegScheme
	{
		private static final long serialVersionUID = 1L;
		@Override
		protected float regLoss_(LearningPlan plan, ParamList paramList)
		{
			float loss = 0;
			for (ParamUnit W : paramList)
				loss += Thrust.square_sum(W.data());
			return 0.5f * loss * plan.reg;
		}

		@Override
		public void regGradUpdate_(LearningPlan plan, ParamUnit W)
		{
			GpuBlas.scale(W.data(), 1 - plan.lr * plan.reg);
		}
	}
	/**
	 * @return Preset scheme: L-2 square-sum regularizer
	 */
	public static RegScheme l2Scheme()
	{
		return new L2RegScheme();
	}
}
