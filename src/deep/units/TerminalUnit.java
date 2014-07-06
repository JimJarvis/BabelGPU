package deep.units;

import gpu.*;

import java.util.ArrayList;
import java.util.Collections;

public abstract class TerminalUnit extends PureComputeUnit
{
	/**
	 * Gold standard, normally the correct labels -> sparse 1-0 matrix
	 */
	public InletUnit inlet; // will give us gold labels
	protected float result = 0;
	protected ArrayList<ParamUnit> wList = null;
	
	public TerminalUnit(String name, InletUnit inlet)
	{
		super(name, 1);
		this.inlet = inlet;
	}
	
	@Override
	public void setup()
	{
		setupLink();
		getParams();
	}
	
	/**
	 * 'final' ensures that subclass cannot directly override forward()
	 * but must implement forward_()
	 */
	@Override
	public final void forward()
	{
		inlet.nextGold();
		forward_();
	}
	/**
	 * Hack: ensure that you call super.forward() first
	 */
	protected abstract void forward_();
	
	/**
	 * @return all paramUnit from previous ParamComputeUnit, in forward order
	 */
	public ArrayList<ParamUnit> getParams()
	{
		if (wList != null) 	return wList;
		
		wList = new ArrayList<>();
		ComputeUnit unitptr = this.prev;
		while (unitptr != null)
		{
			if (unitptr instanceof ParamComputeUnit)
				wList.add(((ParamComputeUnit) unitptr).W);
			unitptr = unitptr.prev;
		}
		Collections.reverse(wList);
		return wList;
	}
	
	public float getResult() {	return result / learningPlan.totalTrainSize;	}
	
	public float update(float update)
	{
		this.learningPlan.curTrainSize += input.batchSize();
		return this.result += update;
	}
	
	private FloatMat tmp_data_sqs[];  // hold temp squared values
	public void updateReg()
	{
		// init once
		if (tmp_data_sqs == null)
		{
			tmp_data_sqs = new FloatMat[wList.size()];
			int i = 0;
			for (ParamUnit w : wList)
				tmp_data_sqs[i ++] = new FloatMat(w.data);
		}
		// L2 regularizer
		float update = 0;
		for (int i = 0; i < wList.size(); i++)
		{
            Thrust.square(wList.get(i).data, tmp_data_sqs[i]);
			update += tmp_data_sqs[i].sum();
		}
		this.result += 0.5 * update * learningPlan.reg;
	}
	
	public void clearResult() { this.result = 0; }
}
