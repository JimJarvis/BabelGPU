package deep.units;

import gpu.*;

public class SigmoidUnit extends PureComputeUnit
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
	public void backward()
	{
		Thrust.sigmoid_deriv(output.data, input.gradient);
		GpuBlas.dotMult(output.gradient, input.gradient);
	}

}
