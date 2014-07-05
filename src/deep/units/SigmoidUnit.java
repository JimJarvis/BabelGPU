package deep.units;

import gpu.*;

public class SigmoidUnit extends ComputeUnit
{
	public SigmoidUnit(String name)
	{
		super(name);
		// newDim should be the same as the dim from the last layer
		this.outDim = prev != null ?
				prev.outDim : input.dim();
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
