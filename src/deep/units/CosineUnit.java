package deep.units;

import gpu.*;

public class CosineUnit extends ElementComputeUnit
{
	public CosineUnit(String name, InletUnit inlet, boolean hasBias, float scalor)
	{
		super(name, inlet, hasBias, scalor);
	}

	public CosineUnit(String name, InletUnit inlet, float scalor)
	{
		super(name, inlet, scalor);
	}
	
	public CosineUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	public void forward_element(float scalor)
	{
		Thrust.cos(input.data(), output.data(), 1, 0, scalor);
	}

	@Override
	public void backward_element()
	{
		FloatMat grad = input.gradient();
		Thrust.sin(input.data(), grad, 1, 0, -1);
	}

}
