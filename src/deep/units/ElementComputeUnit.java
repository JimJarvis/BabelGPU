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
	 * @param hasBias see ComputeUnit
	 */
	public ElementComputeUnit(String name, boolean hasBias)
	{
		super(name, -1, hasBias);
	}
	
	public ElementComputeUnit(String name)
	{
		this(name, true);
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
    		if (hasBias)
    			input.gradient.fillRow(0, -1);
		}
	}
	
	public abstract void backward_element();
}
