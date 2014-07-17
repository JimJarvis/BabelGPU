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
	public void forward_element()
	{
		Thrust.cos(input.data(), output.data());
	}

	@Override
	public void backward_element()
	{
		Thrust.sin(input.data(), input.gradient());
		GpuBlas.scale(input.gradient(), -1);
	}

}
