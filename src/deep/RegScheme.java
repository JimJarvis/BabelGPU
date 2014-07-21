package deep;

import gpu.FloatMat;
import gpu.Thrust;

import java.util.ArrayList;

import utils.PP;
import deep.units.ParamUnit;

/**
 * Regularization scheme
 */
public abstract class RegScheme extends LearningPlan.Scheme
{
	/**
	 * Main interface to outside
	 */
	public final float regularize()
	{
		DeepNet net = plan.net;
		return net.doesCalcLoss() && plan.hasReg() ?
				regularize_(plan, net.getParams()) : 0;
	}
	
	/**
	 * @return regularization loss
	 * @see #regularize(LearningPlan)
	 */
	protected abstract float regularize_(LearningPlan plan, ParamList paramList);
	
	/**
	 * @return Preset scheme: L-2 regularizer
	 */
	public static RegScheme squareSumScheme()
	{
		return new RegScheme()
		{
			@Override
			protected float regularize_(LearningPlan plan, ParamList paramList)
			{
				// L2 regularizer
				float loss = 0;
				for (ParamUnit W : paramList)
					loss += Thrust.square_sum(W.data());
				return 0.5f * loss * plan.reg;
			}
		};
	}
}
