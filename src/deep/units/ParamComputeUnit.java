package deep.units;

import deep.*;

public abstract class ParamComputeUnit extends ComputeUnit
{
	public Initializer wInitializer = Initializer.DUMMY;
	public ParamUnit W;
	
	public ParamComputeUnit(String name, int newDim, Initializer wInitializer)
	{
		super(name, newDim);
		this.wInitializer = wInitializer;
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
		this.wInitializer.init(W);
	}
}
