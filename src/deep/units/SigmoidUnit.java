package deep.units;

import gpu.*;

public class SigmoidUnit extends ElementComputeUnit
{
	private static final long serialVersionUID = 1L;

	public SigmoidUnit(String name, InletUnit inlet, boolean hasBias, float scalor)
	{
		super(name, inlet, hasBias, scalor);
	}

	public SigmoidUnit(String name, InletUnit inlet, float scalor)
	{
		super(name, inlet, scalor);
	}
	
	public SigmoidUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	public void forward_element(float scalor)
	{
		Thrust.sigmoid(input.data(), output.data(), 1, 0, scalor);
	}

	@Override
	public void backward_element()
	{
		// Because the forward method produces y = scalor * sigmoid
		// so its gradient is [y' = y/scalor * (1 - y/scalor)] * scalor
		Thrust.sigmoid_deriv(output.data(), input.gradient(), 1f/scalor, 0, 1);
	}

}
