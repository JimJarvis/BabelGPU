package deep.units;

import deep.*;

public abstract class ParamComputeUnit extends ComputeUnit
{
	private static final long serialVersionUID = 1L;
	public Initializer initer = Initializer.dummyIniter;
	public ParamUnit W;
	
	/**
	 * @param outDim output dimension from this computing unit
	 * @param initer parameter W initializer
	 * @param hasBias see ComputeUnit
	 */
	public ParamComputeUnit(String name, InletUnit inlet, int outDim, boolean hasBias, Initializer initer)
	{
		super(name, inlet, outDim, hasBias);
		this.initer = initer;
	}
	
	/**
	 * @param outDim output dimension from this computing unit
	 * @param initer parameter W initializer
	 */
	public ParamComputeUnit(String name, InletUnit inlet, int outDim, Initializer initer)
	{
		this(name, inlet, outDim, true, initer);
	}
	
	/**
	 * Default: init W to all zero
	 */
	public ParamComputeUnit(String name, InletUnit inlet, int outDim)
	{
		this(name, inlet, outDim, true, Initializer.fillIniter(0));
	}
	
	@Override
	public void setup()
	{
		super.setup();
		setupW();
	}
	
	/**
	 * Construct and setup the parameter
	 * @see #reInit()
	 */
	protected abstract void setupW();
	
	/**
	 * Re-initialize W
	 */
	public void reInit()
	{
		this.initer.init(W);
	}
}
