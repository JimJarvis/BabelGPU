package deep.units;

import deep.*;

public abstract class ParamComputeUnit extends ComputeUnit
{
	public Initializer initer = Initializer.dummyIniter;
	public ParamUnit W;
	
	/**
	 * @param newDim output dimension from this computing unit
	 * @param initer parameter W initializer
	 */
	public ParamComputeUnit(String name, int newDim, Initializer initer)
	{
		super(name, newDim);
		this.initer = initer;
	}
	
	@Override
	public void setup()
	{
		super.setup();
		setupOutput();
		setupW();
	}
	
	protected void setupW()
	{
		this.W = new ParamUnit("W[" + this.name + "]", this, outDim, input.dim());
		reInit();
		if (debug)
			this.W.initGradient();
	}
	
	/**
	 * Re-initialize W
	 */
	public void reInit()
	{
		this.initer.init(W);
	}
}
