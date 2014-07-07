package deep.units;

import gpu.*;

import java.util.ArrayList;
import java.util.Collections;

public abstract class TerminalUnit extends ComputeUnit
{
	/**
	 * Gold standard, normally the correct labels -> sparse 1-0 matrix
	 */
	public InletUnit inlet; // will give us gold labels
	/**
	 * Pure loss function
	 */
	protected float lossPure = 0;
	/**
	 * Loss due to regularization
	 */
	protected float lossReg = 0;
	/**
	 * Parameter list, in forward order
	 */
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
	 * but must implement forward_terminal()
	 */
	@Override
	public final void forward()
	{
		inlet.nextGold();
		if (learningPlan.hasReg())
			updateLossReg();
		
		// Will be implemented by subclasses
		forward_terminal();
	}
	/**
	 * Hack: ensure that you call super.forward() first
	 */
	protected abstract void forward_terminal();
	
	/**
	 * @return all paramUnit from previous ParamComputeUnit, in forward order
	 */
	public ArrayList<ParamUnit> getParams()
	{
		// Make sure we get the latest list of params
		// a non-empty wList with null entries means the parameters aren't set yet
		if (wList != null && wList.size() != 0 && wList[0] != null)
			return wList;
		
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
	
	public float lossTotal() { return lossPure() + lossReg() ;	}
	
	/**
	 * Normalized by curTrainSize (number of training samples seen so far)
	 */
	public float lossPure()
	{
		return this.lossPure / learningPlan.curTrainSize;
	}
	
	public float lossReg() {	return this.lossReg;	}
	
	
	public float updateLossPure(float update)
	{
		this.learningPlan.curTrainSize += input.batchSize();
		return this.lossPure += update;
	}
	
	private FloatMat tmp_data_sqs[];  // hold temp squared values
	public void updateLossReg()
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
		float loss = 0;
		for (int i = 0; i < wList.size(); i++)
		{
            Thrust.square(wList.get(i).data, tmp_data_sqs[i]);
			loss += tmp_data_sqs[i].sum();
		}
		this.lossReg += 0.5 * loss * learningPlan.reg;
	}
	
	public void clearLoss()
	{ 
		this.lossPure = 0;
		this.lossReg = 0;
	}
}
