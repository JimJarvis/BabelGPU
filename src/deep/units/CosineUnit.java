package deep.units;

import gpu.*;

public class CosineUnit extends ElementComputeUnit
{
	public CosineUnit(String name, boolean hasBias, float scalor)
	{
		super(name, hasBias, scalor);
	}

	public CosineUnit(String name, float scalor)
	{
		super(name, scalor);
	}
	
	public CosineUnit(String name)
	{
		super(name);
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
