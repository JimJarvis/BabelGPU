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
		super(name);
		this.outDim = prev != null ?
				prev.outDim : input.dim();
	}
}
