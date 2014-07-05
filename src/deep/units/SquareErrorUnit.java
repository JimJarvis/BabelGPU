package deep.units;

import gpu.*;

// terminal unit
public class SquareErrorUnit extends TerminalUnit
{
	private FloatMat y_minus_input_squared; // temp cache

	public SquareErrorUnit(String name, DataUnit y)
	{
		super(name, y);
		y_minus_input_squared= new FloatMat(y.data);
	}

	@Override
	public void forward()
	{
		GpuBlas.add(input.data, y.data, input.gradient, 1, -1);
		Thrust.square(input.gradient, y_minus_input_squared);
		update(y_minus_input_squared.linear(0.5f, 0).sum());
	}

	@Override
	public void backward()
	{
		// Already done by forward()
	}

}
