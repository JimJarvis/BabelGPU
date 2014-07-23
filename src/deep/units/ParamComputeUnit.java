package deep.units;

import deep.*;

public abstract class ParamComputeUnit extends ComputeUnit
{
	private static final long serialVersionUID = 1L;
	public Initializer initer = Initializer.dummyIniter;
	public ParamUnit W;
	// serialization
	protected int paramSaveMode = DataUnit.SAVE_DATA;
	
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
	
	public final void setupW()
	{
		// Either it's the first-time setup, or we are loading from disk and we aren't saving anything
		if (W == null || !W.doesSaveData())
		{
			setupW_();
		}
		this.W.setSaveMode(paramSaveMode);
		if (debug)
			this.W.initGradient();
		else
			this.W.setDummyGradient();
	}
	
	/**
	 * Construct and setup the parameter
	 * @see #reInit()
	 * @see #setupW()
	 */
	protected abstract void setupW_();
	
	/**
	 * Serialization 
	 * Default SAVE_DATA
	 */
	public void setParamSaveMode(int saveMode)
	{
		this.paramSaveMode = saveMode;
		if (this.W != null)
			this.W.setSaveMode(this.paramSaveMode);
	}
	
	/**
	 * Re-initialize W
	 */
	public void reInit()
	{
		this.initer.init(W);
	}
}
