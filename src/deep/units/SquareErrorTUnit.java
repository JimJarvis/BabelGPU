package deep.units;

import gpu.*;

// terminal unit
public class SquareErrorTUnit extends TerminalUnit
{
	private static final long serialVersionUID = 1L;

	public SquareErrorTUnit(String name, InletUnit inlet, boolean hasBias)
	{
		super(name, inlet, hasBias);
	}

	public SquareErrorTUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	protected float forward_terminal(boolean doesCalcLoss)
	{
		float norm = super.batchNormalizer();
		FloatMat data = input.data();
		FloatMat grad = input.gradient();
		if (input.hasGradient())
		{
			// This is actually the backward step:
			GpuBlas.add(data, inlet.goldMat, grad, norm, -norm);

			if (doesCalcLoss)
				// we give back what we divide too much
				// This amount will be used to update lossPure
				return grad.square_sum() / (2 * norm * norm);
		}
		else // if input doesn't have gradient, mutate input.data
			if (doesCalcLoss)
				return GpuBlas.add(data, inlet.goldMat, data, norm, -norm).square_sum()
						/ (2 * norm * norm);
		
		return 0;  // if !doesCalcLoss
	}

	@Override
	public void backward()
	{
    	// backward is already done by forward
	}
}
