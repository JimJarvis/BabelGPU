package deep.units;

import gpu.*;

public class SparseCrossEntropyUnit extends TerminalUnit
{
	public SparseCrossEntropyUnit(String name, InletUnit inlet, boolean hasBias)
	{
		super(name, inlet, hasBias);
	}

	public SparseCrossEntropyUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	private FloatMat tmp_outLogProb;
	
	@Override
	protected float forward_terminal()
	{
		int batch = input.batchSize();

		if (tmp_outLogProb == null)
			tmp_outLogProb = new FloatMat(batch, 1);

		return 
			- Thrust.batch_softmax_at_label(input.data(), tmp_outLogProb, inlet.goldLabels, hasBias);
	}

	@Override
	public void backward()
	{
		// Gradient = 1/batch * (y - id)
		Thrust.batch_softmax_minus_id(input.data(), input.gradient(), inlet.goldLabels, hasBias);
		GpuBlas.scale(input.gradient(), super.batchNormalizer());
	}

}
