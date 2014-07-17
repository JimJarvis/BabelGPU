package deep.units;

import deep.DeepException;
import gpu.Thrust;

/**
 * Debug only
 */
public class SumTUnit extends TerminalUnit
{
	public SumTUnit(String name, InletUnit inlet, boolean hasBias)
	{
		super(name, inlet, hasBias);
	}

	public SumTUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}
	
	@Override
	public void setup()
	{
		if (!debug)
			throw new DeepException(
					"The implementation is very inefficient. Please run under debug mode only.");
		super.setup();
	}

	@Override
	protected float forward_terminal(boolean doesCalcLoss)
	{
		if (!doesCalcLoss)	return 0;
		return input.data().sum();
	}

	@Override
	public void backward()
	{
		if (input.hasGradient())
    		Thrust.fill(input.gradient(), super.batchNormalizer());
	}

}
