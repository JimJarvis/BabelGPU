package deep.units;

import deep.LearningPlan;
import gpu.*;

public abstract class ComputeUnit extends Unit
{
	public ComputeUnit next = null;
	public ComputeUnit prev = null;
	protected int outDim;
	// Do we include bias units?
	protected boolean hasBias;
	// Do we store input/output data separately?
	protected boolean mergeIO = false;
	protected LearningPlan learningPlan;

	/**
	 * ALWAYS equal to prev.output
	 */
	public DataUnit input;
	/**
	 * ALWAYS equal to next.input
	 */
	public DataUnit output;
	
	public ComputeUnit(String name)
	{
		super(name);
	}
	
	/**
	 * @param outDim the dimension of the output (transformed) data
	 * @param hasBias if true, the actual outDim will be your input + 1
	 */
	public ComputeUnit(String name, int outDim, boolean hasBias)
	{
		super(name);
		this.hasBias = hasBias;
		this.outDim = hasBias ? outDim + 1 : outDim;
	}
	
	/**
	 * Default hasBias = true, the actual outDim will be your input + 1
	 */
	public ComputeUnit(String name, int outDim)
	{
		this(name, outDim, true);
	}
	
	/**
	 * Forward propagation abstraction
	 */
	public abstract void forward(); 
	
	/**
	 * Backward propagation abstraction
	 */
	public abstract void backward();
	
	public void setup()
	{
		setupLink();
		setupOutput();
	}
	
	protected void setupLink()
	{
		if (prev != null)
    		this.input = prev.output;
	}
	
	protected void setupOutput()
	{
		if (mergeIO) // use memory efficiently
			this.output = this.input;
		else
		{
			this.output = new DataUnit("out[" + this.name + "]", this, new FloatMat(outDim, input.batchSize()));
			this.output.initGradient();
		}
	}
	
	/**
	 * Do this BEFORE setup
	 */
	public void setLearningPlan(LearningPlan learningPlan)
	{
		this.learningPlan = learningPlan;
	}
	
	/**
	 * Needs to be called BEFORE setup() !!
	 */
	public void setBias(boolean hasBias)
	{
		if (this.hasBias != hasBias)
		{
			if (this.hasBias) // switch bias off
				-- outDim;
			else
				++ outDim;
			this.hasBias = hasBias;
		}
	}
	
	/**
	 * @return 'real' outDim: if hasBias, outDim-1
	 */
	public int biaslessOutdim()
	{
		return this.outDim - (hasBias ? 1 : 0);
	}
	
	/**
	 * 'mergeIO' flag: whether or not 'input' and 'output' will be distinct memory places. 
	 * default false. If set to true, 'input' and 'output' will essentially be the same, 
	 * so changes will be made in-place (intrusively). 
	 * Turn to true for cases where input.data would never be needed again. Optimize memory usage
	 * Needs to be called BEFORE setup()
	 */
	public void setMergeIO(boolean mergeIO) { this.mergeIO = mergeIO; }
}
