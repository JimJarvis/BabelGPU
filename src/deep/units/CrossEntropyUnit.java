package deep.units;

public class CrossEntropyUnit extends TerminalUnit
{
	public CrossEntropyUnit(String name, InletUnit inlet, boolean hasBias)
	{
		super(name, inlet, hasBias);
	}

	public CrossEntropyUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	protected float forward_terminal()
	{
		
	}

	@Override
	public void backward()
	{
	}

}
