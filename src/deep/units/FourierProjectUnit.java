package deep.units;

import gpu.*;
import deep.DeepException;
import deep.Initializer;

/**
 * Rahami-Recht random Fourier projection features
 * The input must have an extra row for bias
 */
public class FourierProjectUnit extends ComputeUnit
{
	private static final long serialVersionUID = 1L;
	// This is NOT a learned parameter: it's a fixed random projection matrix
	private ParamUnit projector;
	private Initializer projIniter; // projection
	
	/**
	 * 	Default: assume input has an extra row of 1 (bias unit), 
	 * the projection matrix will have an extra row all zeros, except for the last element which is a 1.
	 * in this way ProjMat * Input will preserve the extra row of 1
	 * 'hasBias' must be true because the last column U[0, 2*pi] needs to be added
	 * @param projIniter use a projKernelIniter instead of a pure distrIniter
	 */
	public FourierProjectUnit(String name, InletUnit inlet, int outDim, Initializer projIniter)
	{
		super(name, inlet, outDim, true);
		this.projIniter = projIniter;
	}
	
	@Override
	public void setup()
	{
		if (!hasBias && !debug)
			throw new DeepException("FourierProjectUnit requires that hasBias is set to true.");
			
		super.setup();
		// Leave the 'parent' ParamComputeUnit null
		projector = new ParamUnit(
				"Param[projector]#" + this.name, 
				outDim, 
				input.dim());
		projector.setNoGradient();
		reInitProjector();
	}
	
	/**
	 * Explicitly reinitialize the projector
	 */
	public void reInitProjector()
	{
		this.projIniter.setBias(hasBias); // hasBias is always true
		this.projIniter.init(projector);
	}
	
	public FloatMat getProjection()
	{
		return projector.data();
	}

	@Override
	public void forward()
	{
		if (hasBias)   // hasBias must be true for the projection to work
    		input.data().fillLastRow1();
		GpuBlas.mult(projector.data(), input.data(), output.data());
	}

	@Override
	public void backward()
	{
		// update input.gradient() only when necessary 
		// Don't upgrade the gradient of the input layer, of course
		if (input.hasGradient())
		{
			GpuBlas.mult(projector.data().transpose(), output.gradient(), input.gradient());
    		if (debug && hasBias)
    			input.gradient().fillLastRow0();
		}
	}
}
