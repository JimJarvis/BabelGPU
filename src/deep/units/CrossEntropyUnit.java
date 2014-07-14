package deep.units;

import deep.*;
import utils.*;
import gpu.*;

public class CrossEntropyUnit extends TerminalUnit
{
	public CrossEntropyUnit(String name, InletUnit inlet, boolean hasBias)
	{
		super(name, inlet, hasBias);
	}

	public CrossEntropyUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	private FloatMat tmp_softmax = null;
	
	@Override
	protected float forward_terminal()
	{
		Thrust.batch_softmax(input.data(), input.gradient());

		if (tmp_softmax == null)
			tmp_softmax = new FloatMat(input.data());
		Thrust.log(input.gradient(), tmp_softmax);

		// Cross entropy: - t * log(y) where 't' is target value, 'y' is actual output
		GpuBlas.dotMult(inlet.goldMat, tmp_softmax);
		return - tmp_softmax.sum();
	}

	@Override
	public void backward()
	{
		// Gradient = 1/batch * (y - t)
		float norm = super.batchNormalizer();
		GpuBlas.add(input.gradient(), inlet.goldMat, input.gradient(), norm, -norm);

		// NOTE: the gradient is incorrect if each column of 'gold' doesn't sum up to 1
		// Debug ONLY
		if (debug)
			for (int c = 0; c < inlet.goldMat.col; c++)
				if (! CpuUtil.equal(inlet.goldMat.createColOffset(c, c+1).sum(), 1, 1e-5f))
					throw new DeepException("Each column of the goldMat must sum up to 1\n"
							+ "Otherwise the softmax-cross entropy gradient is incorrect.");
	}

}
