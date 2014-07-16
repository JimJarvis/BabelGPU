package deep.units;

import deep.DeepException;

/**
 * For forward only calculations.
 * Isn't actually a neural network. Doesn't support back-prop
 */
public class ForwardOnlyUnit extends TerminalUnit
{
	public ForwardOnlyUnit(String name)
	{
		super(name, null);
	}
	
	@Override
	public void setup()
	{
		super.setup();
	}
	
	@Override
	public final void forward()
	{
		
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
