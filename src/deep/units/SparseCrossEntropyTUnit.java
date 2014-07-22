package deep.units;

import gpu.*;

public class SparseCrossEntropyTUnit extends TerminalUnit
{
	private static final long serialVersionUID = 1L;

	public SparseCrossEntropyTUnit(String name, InletUnit inlet, boolean hasBias)
	{
		super(name, inlet, hasBias);
	}

	public SparseCrossEntropyTUnit(String name, InletUnit inlet)
	{
		super(name, inlet);
	}

	private FloatMat tmp_outLogProb;
	
	@Override
	protected float forward_terminal(boolean doesCalcLoss)
	{
		if (!doesCalcLoss)	return 0;
		
		int batch = inlet.batchSize;
		if (tmp_outLogProb == null)
			tmp_outLogProb = new FloatMat(batch, 1);

		return 
			- Thrust.batch_softmax_at_label(input.data(), tmp_outLogProb, inlet.goldLabels, hasBias);
	}

	@Override
	public void backward()
	{
		if (input.hasGradient())
		{
    		// Gradient = 1/batch * (y - id)
			FloatMat grad = input.gradient();
    		Thrust.batch_softmax_minus_id(input.data(), grad, inlet.goldLabels, hasBias);
    		GpuBlas.scale(grad, super.batchNormalizer());
		}
	}

}
