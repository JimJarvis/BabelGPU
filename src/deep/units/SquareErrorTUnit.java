package deep.units;

import gpu.*;

// terminal unit
public class SquareErrorTUnit extends TerminalUnit
{
	public SquareErrorTUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	private FloatMat tmp_y_minus_input_sq = null; // temp cache

	@Override
	public float forward_terminal()
	{
		if (tmp_y_minus_input_sq == null)
			tmp_y_minus_input_sq = new FloatMat(input.data());
		float norm = super.batchNormalizer();
		
		// This is actually the backward step:
		GpuBlas.add(input.data(), inlet.goldMat, input.gradient(), norm, -norm);
		Thrust.square(input.gradient(), tmp_y_minus_input_sq);
		
        // we give back what we divide too much
		// This amount will be used to update lossPure
		return tmp_y_minus_input_sq.sum() / (2 * norm * norm);
	}

	@Override
	public void backward()
	{
    	// backward is already done by forward
	}
}
