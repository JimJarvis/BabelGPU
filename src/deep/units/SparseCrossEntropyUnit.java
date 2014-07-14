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

	private FloatMat tmp_outProb;
	
	@Override
	protected float forward_terminal()
	{
		int batch = input.batchSize();

		if (tmp_outProb == null)
			tmp_outProb = new FloatMat(batch, 1);
		else if (batch != tmp_outProb.col)
			tmp_outProb = tmp_outProb.createColOffset(0, batch);

		Thrust.batch_softmax_at_label(input.data(), tmp_outProb, inlet.goldLabels);
		
		return - Thrust.log_sum(tmp_outProb);
	}

	@Override
	public void backward()
	{
		// Gradient = 1/batch * (y - id)
		Thrust.batch_softmax_minus_id(input.data(), input.gradient(), inlet.goldLabels);
		GpuBlas.scale(input.gradient(), super.batchNormalizer());
	}

}
