package deep.units;

import gpu.GpuBlas;

/**
 * Element-wise transformation
 * input dimensions == output dimensions
 */
public abstract class ElementComputeUnit extends ComputeUnit
{
	protected float scalor = 1;
	
	/**
	 * outDim always equal to prev.outDim or input.dim
	 * @param hasBias see ComputeUnit
	 * @param scalor default = 1 multiply every element by scalor to 'output'
	 */
	public ElementComputeUnit(String name, boolean hasBias, float scalor)
	{
		super(name, -1, hasBias);
		this.scalor = scalor;
	}
	
	public ElementComputeUnit(String name, float scalor)
	{
		this(name, true, scalor);
	}
	
	/**
	 * Default scalor = 1
	 */
	public ElementComputeUnit(String name)
	{
		this(name, true, 1);
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
	 * 'final' ensures that subclass cannot directly override forward()
	 * but must implement forward_element()
	 */
	@Override
	public final void forward()
	{
		forward_element();
		if (scalor != 1)
    		output.data.linear(scalor, 0);
	}
	
	public abstract void forward_element();
	
	/**
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
    		if (scalor != 1)
    			input.gradient.linear(scalor, 0);
    		if (hasBias)
    			input.gradient.fillRow(0, -1);
		}
	}
	
	public abstract void backward_element();
}
