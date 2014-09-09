package deep.units;

import gpu.*;

public class RectifiedUnit extends ElementComputeUnit
{
	private static final long serialVersionUID = 1L;

	public RectifiedUnit(String name, InletUnit inlet, boolean hasBias, float scalor)
	{
		super(name, inlet, hasBias, scalor);
	}

	public RectifiedUnit(String name, InletUnit inlet, float scalor)
	{
		super(name, inlet, scalor);
	}
	
	public RectifiedUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	public void forward_element(float scalor)
	{
		Thrust.rectified_linear(input.data(), output.data(), 1, 0, scalor);
	}

	@Override
	public void backward_element()
	{
		Thrust.rectified_linear_deriv(input.data(), input.gradient());
	}

}