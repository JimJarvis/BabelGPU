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
	
	/**
	 * 	Default: assume input has an extra row of 1 (bias unit), 
	 * the projection matrix will have an extra row all zeros, except for the last element which is a 1.
	 * in this way ProjMat * Input will preserve the extra row of 1
	 * Default: hasBias is true
	 * @param projIniter use a projKernelIniter instead of a pure distrIniter
	 */
	public FourierProjectUnit(String name, int outDim, Initializer projIniter)
	{
		super(name, outDim, true);
		this.projIniter = projIniter;
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
		return projector.data();
	}

	@Override
	public void forward()
	{
		if (hasBias)
		// The last row will have all ones
			input.data().fillLastRow1();
		GpuBlas.mult(projector.data(), input.data(), output.data());
	}

	@Override
	public void backward()
	{
		// update input.gradient() only when necessary 
		// Don't upgrade the gradient of the input layer, of course
		if (input.hasGradient())
			GpuBlas.mult(projector.data().transpose(), output.gradient(), input.gradient());
	}
}
