package deep.units;

import java.util.ArrayList;

public abstract class TerminalUnit extends PureComputeUnit
{
	/**
	 * Gold standard, normally the correct labels -> sparse 1-0 matrix
	 */
	public InletUnit inlet; // will give us gold labels
	protected float result = 0;
	protected ArrayList<ParamUnit> wList;
	
	public TerminalUnit(String name, InletUnit inlet)
	{
		super(name, 1);
		this.inlet = inlet;
	}
	
	@Override
	public void setup()
	{
		setupLink();
		collectParams();
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
	 * @return all paramUnit from previous ParamComputeUnit
	 */
	public ArrayList<ParamUnit> collectParams()
	{
		wList = new ArrayList<>();
		ComputeUnit unitptr = this.prev;
		while (unitptr != null)
		{
			if (unitptr instanceof ParamComputeUnit)
				wList.add(((ParamComputeUnit) unitptr).W);
			unitptr = unitptr.prev;
		}
		return wList;
	}
	
	public float getResult() {	return result;	}
	
	public float update(float update)
	{
		this.learningPlan.curTrainSize += input.batchSize();
		return this.result += update;
	}
	
	public void updateReg()
	{
		// L2 regularizer
		float update = 0;
		for (ParamUnit w : wList)
			update += w.data.square().sum();
		this.result += update / learningPlan.totalTrainSize;
	}
}
