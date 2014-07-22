package deep.units;

import deep.*;
import utils.*;
import gpu.*;

public class CrossEntropyTUnit extends TerminalUnit
{
	private static final long serialVersionUID = 1L;

	public CrossEntropyTUnit(String name, InletUnit inlet, boolean hasBias)
	{
		super(name, inlet, hasBias);
	}

	public CrossEntropyTUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	private FloatMat tmp_softmax = null;
	
	@Override
	protected float forward_terminal(boolean doesCalcLoss)
	{
		FloatMat data = input.data();
		FloatMat grad = input.gradient();
		if (input.hasGradient())
		{
    		Thrust.batch_softmax(data, grad, hasBias);
    
    		if (doesCalcLoss)
    		{
    			if (tmp_softmax == null)
    				tmp_softmax = new FloatMat(data);

    			Thrust.log(grad, tmp_softmax);
    			if (hasBias)	tmp_softmax.fillLastRow0(); // because the last row would be log(0) -> NaN

    			// Cross entropy: - t * log(y) where 't' is target value, 'y' is actual output
    			GpuBlas.dotMult(inlet.goldMat, tmp_softmax);
    			return - tmp_softmax.sum();
    		}
		}
		else // If input doesn't have gradient, mutate input.data
		{
			Thrust.batch_softmax(data, hasBias);
			
			if (doesCalcLoss)
			{
    			Thrust.log(data);
    			if (hasBias)	data.fillLastRow0();
    			GpuBlas.dotMult(inlet.goldMat, data);
    			return - data.sum();
			}
		}
		return 0;  // if !doesCalcLoss
	}

	@Override
	public void backward()
	{
		if (input.hasGradient())
		{
    		// Gradient = 1/batch * (y - t)
    		float norm = super.batchNormalizer();
    		FloatMat grad = input.gradient();
    		GpuBlas.add(grad, inlet.goldMat, grad, norm, -norm);
    
    		// NOTE: the gradient is incorrect if each column of 'gold' doesn't sum up to 1
    		// Debug ONLY
    		if (debug)
    			for (int c = 0; c < inlet.goldMat.col; c++)
    				if (! CpuUtil.equal(inlet.goldMat.createColOffset(c).sum(), 1, 1e-5f))
    					throw new DeepException("Each column of the goldMat must sum up to 1\n"
    							+ "Otherwise the softmax-cross entropy gradient is incorrect.");
		}
	}

}
