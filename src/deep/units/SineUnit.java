package deep.units;

import gpu.*;

public class SineUnit extends ElementComputeUnit
{
	private static final long serialVersionUID = 1L;

	public SineUnit(String name, InletUnit inlet, boolean hasBias, float scalor)
	{
		super(name, inlet, hasBias, scalor);
	}

	public SineUnit(String name, InletUnit inlet, float scalor)
	{
		super(name, inlet, scalor);
	}
	
	public SineUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	public void forward_element(float scalor)
	{
		Thrust.sin(input.data(), output.data(), 1, 0, scalor);
	}

	@Override
	public void backward_element()
	{
		Thrust.cos(input.data(), input.gradient());
	}

}
