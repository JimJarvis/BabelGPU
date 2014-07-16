package deep.units;

import deep.DeepException;

/**
 * For forward only calculations.
 * Isn't actually a neural network. Doesn't support back-prop
 */
public class ForwardOnlyTUnit extends TerminalUnit
{
	public ForwardOnlyTUnit(String name)
	{
		super(name, null);
	}
	
	@Override
	public void setup()
	{
		super.setup();
		this.output = this.input;
	}
	
	@Override
	public final void forward()
	{ 
		super.updateLearningPlan();
	}

	@Override
	protected float forward_terminal() { return 0; }

	@Override
	public final void backward()
	{
		throw new DeepException(
				"Forward-only terminal doesn't support back-prop. "
				+ "It's meant for pure calculation only.");
	}

}
