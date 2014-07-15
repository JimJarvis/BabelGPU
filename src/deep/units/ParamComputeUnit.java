package deep.units;

import deep.*;

public abstract class ParamComputeUnit extends ComputeUnit
{
	public Initializer initer = Initializer.dummyIniter;
	public ParamUnit W;
	
	/**
	 * @param outDim output dimension from this computing unit
	 * @param initer parameter W initializer
	 * @param hasBias see ComputeUnit
	 */
	public ParamComputeUnit(String name, int outDim, boolean hasBias, Initializer initer)
	{
		super(name, outDim, hasBias);
		this.initer = initer;
	}
	
	/**
	 * @param outDim output dimension from this computing unit
	 * @param initer parameter W initializer
	 */
	public ParamComputeUnit(String name, int outDim, Initializer initer)
	{
		this(name, outDim, true, initer);
	}
	
	/**
	 * Default: init W to all zero
	 */
	public ParamComputeUnit(String name, int outDim)
	{
		this(name, outDim, true, Initializer.fillIniter(0));
	}
	
	@Override
	public void setup()
	{
		super.setup();
		setupW();
	}
	
	protected void setupW()
	{
		this.initer.setBias(hasBias);
		this.W = new ParamUnit("W[" + this.name + "]", this);
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
