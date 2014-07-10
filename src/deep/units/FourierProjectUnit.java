package deep.units;

import gpu.*;
import deep.Initializer;

/**
 * Rahami-Recht random Fourier projection features
 */
public class FourierProjectUnit extends ComputeUnit
{
	// This is NOT a learned parameter: it's a fixed random projection matrix
	private ParamUnit projector;
	private Initializer projIniter; // projection
	private boolean addRow1; 
	
	/**
	 * @param if addRow1 is true, outDim += 1
	 * 	Default: assume input has an extra row of 1 (bias unit), 
	 * the projection matrix will have an extra row all zeros, except for the last element which is a 1.
	 * in this way ProjMat * Input will preserve the extra row of 1
	 */
	public FourierProjectUnit(String name, int outDim, Initializer projIniter, boolean addRow1)
	{
		super(name, addRow1 ? outDim + 1 : outDim);
		this.projIniter = projIniter;
		this.addRow1 = addRow1;
	}
	
	/**
	 * Default: addRow1 is true
	 */
	public FourierProjectUnit(String name, int outDim, Initializer projIniter)
	{
		this(name, outDim, projIniter, true);
	}
	
	@Override
	public void setup()
	{
		super.setup();
		// Leave the 'parent' ParamComputeUnit null
		projector = new ParamUnit("projector["+this.name+"]", outDim, input.dim());
		projector.setNoGradient();
		reInitProjector();
	}
	
	/**
	 * Explicitly reinitialize the projector
	 */
	public void reInitProjector()
	{
		this.projIniter.init(projector);
	}
	
	public FloatMat getProjection()
	{
		return projector.data;
	}

	@Override
	public void forward()
	{
		GpuBlas.mult(projector.data, input.data, output.data);
		// The last row will have all ones
		
	}

	@Override
	public void backward()
	{
		// update input.gradient only when necessary 
		// Don't upgrade the gradient of the input layer, of course
		if (input.hasGradient())
			GpuBlas.mult(projector.data.transpose(), output.gradient, input.gradient);
	}
}
