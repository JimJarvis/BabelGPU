package deep.units;

import gpu.*;

// terminal unit
public class SquareErrorUnit extends TerminalUnit
{
	private FloatMat y_minus_input_squared = null; // temp cache

	public SquareErrorUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	public void forward_()
	{
		if (y_minus_input_squared == null)
			y_minus_input_squared = new FloatMat(input.data);
		GpuBlas.add(input.data, inlet.goldMat, input.gradient, 1, -1);
		Thrust.square(input.gradient, y_minus_input_squared);
		update(y_minus_input_squared.linear(0.5f, 0).sum());
		updateReg();
	}

	@Override
	public void backward()
	{
    	// backward is already done by forward
	}
}
