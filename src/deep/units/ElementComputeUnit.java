package deep.units;

import gpu.GpuBlas;

/**
 * Element-wise transformation
 * input dimensions == output dimensions
 */
public abstract class ElementComputeUnit extends ComputeUnit
{
	/**
	 * outDim always equal to prev.outDim or input.dim
	 */
	public ElementComputeUnit(String name)
	{
		super(name, -1);
	}

	@Override
	public void setup()
	{
		setupLink();
		this.outDim = prev != null ?
				prev.outDim : input.dim();
		setupOutput();
	}
	
	/**
	 * 
	 * 'final' ensures that subclass cannot directly override backward()
	 * but must implement backward_element()
	 */
	@Override
	public final void backward()
	{
		if (input.hasGradient())
		{
			backward_element();
    		GpuBlas.dotMult(output.gradient, input.gradient);
		}
	}
	
	public abstract void backward_element();
}
