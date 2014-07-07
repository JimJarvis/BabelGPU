package deep.units;

import gpu.FloatMat;
import gpu.GpuBlas;
import deep.Initializer;

/**
 * Rahami-Recht random Fourier projection features
 */
public class FourierProjectUnit extends PureComputeUnit
{
	// This is NOT a learned parameter: it's a fixed random projection matrix
	private ParamUnit projector;
	private Initializer projIniter; // projection
	
	public FourierProjectUnit(String name, int outDim, Initializer projIniter)
	{
		super(name, outDim);
		this.projIniter = projIniter;
	}
	
	@Override
	public void setup()
	{
		setupLink();
		setupOutput();
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
