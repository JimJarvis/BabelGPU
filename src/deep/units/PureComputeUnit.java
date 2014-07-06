package deep.units;

public abstract class PureComputeUnit extends ComputeUnit
{
	public PureComputeUnit(String name, int outDim)
	{
		super(name, outDim);
	}
	
	/**
	 * Ctor: outDim the same as input dim from last layer
	 */
	public PureComputeUnit(String name)
	{
		super(name, -1);
	}
	
	@Override
	public void setup()
	{
		super.setup();
		if (outDim < 0) // not initialized, use dim from last layer
    		this.outDim = prev != null ?
    				prev.outDim : input.dim();
		setupOutput();
	}
}
