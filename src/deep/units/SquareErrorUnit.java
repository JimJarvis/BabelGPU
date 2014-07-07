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
		float sizeNorm = 1f/input.batchSize();
		
		// This is actually the backward step:
		GpuBlas.add(input.data, inlet.goldMat, input.gradient, sizeNorm, -sizeNorm);
		Thrust.square(input.gradient, tmp_y_minus_input_sq);
		
        // we give back what we divide too much
		updateLossPure(tmp_y_minus_input_sq.sum() / (2 * sizeNorm * sizeNorm));
		updateLossReg();
	}

	@Override
	public void backward()
	{
    	// backward is already done by forward
	}
}
