package deep.units;

import gpu.FloatMat;
import deep.Initializer;
import deep.LearningPlan;

public abstract class ComputeUnit extends Unit
{
	public ComputeUnit next = null;
	public ComputeUnit prev = null;
	public int outDim;
	protected Initializer wInitializer = Initializer.DUMMY;

	/**
	 * ALWAYS equal to prev.output
	 */
	public DataUnit input;
	/**
	 * ALWAYS equal to next.input
	 */
	public DataUnit output;
	public ParamUnit W;
	
	public LearningPlan learningPlan;
	
	public ComputeUnit(String name)
	{
		super(name);
	}
	
	public ComputeUnit(String name, int newDim, Initializer wInitializer)
	{
		super(name);
		this.outDim = newDim;
		this.wInitializer = wInitializer;
	}
	
	public void setup()
	{
		if (prev != null)
    		this.input = prev.output;
		this.W = new ParamUnit("W[" + this.name + "]", outDim, input.dim());
		this.wInitializer.init(W);
	}
	
	public abstract void forward(); 
	public abstract void backward();
}
