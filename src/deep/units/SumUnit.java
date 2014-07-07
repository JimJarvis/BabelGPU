package deep.units;

import deep.DeepException;
import gpu.Thrust;

/**
 * Debug only
 */
public class SumUnit extends TerminalUnit
{
	public SumUnit(String name, InletUnit inlet)
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
	protected void forward_()
	{
		updateLossPure(input.data.sum());
	}

	@Override
	public void backward()
	{
		Thrust.fill(input.gradient, 1f / input.batchSize());
	}

}