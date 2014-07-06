package deep.units;

import gpu.*;

// terminal unit
public class SquareErrorUnit extends TerminalUnit
{
	private FloatMat tmp_y_minus_input_sq = null; // temp cache

	public SquareErrorUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	@Override
	public void forward_()
	{
		if (tmp_y_minus_input_sq == null)
			tmp_y_minus_input_sq = new FloatMat(input.data);
		GpuBlas.add(input.data, inlet.goldMat, input.gradient, 1, -1);
		Thrust.square(input.gradient, tmp_y_minus_input_sq);
		update(tmp_y_minus_input_sq.linear(0.5f, 0).sum());
		updateReg();
	}

	@Override
	public void backward()
	{
    	// backward is already done by forward
	}
}
