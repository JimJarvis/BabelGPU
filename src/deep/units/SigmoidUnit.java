package deep.units;

import gpu.*;

public class SigmoidUnit extends ElementComputeUnit
{
	public SigmoidUnit(String name)
	{
		super(name);
	}

	@Override
	public void forward()
	{
		Thrust.sigmoid(input.data, output.data);
	}

	@Override
	public void backward_element()
	{
		Thrust.sigmoid_deriv(output.data, input.gradient);
	}

}
