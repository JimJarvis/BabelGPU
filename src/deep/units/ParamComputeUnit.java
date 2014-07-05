package deep.units;

import gpu.FloatMat;
import deep.*;

public abstract class ParamComputeUnit extends ComputeUnit
{
	protected Initializer wInitializer = Initializer.DUMMY;
	public ParamUnit W;
	
	public ParamComputeUnit(String name, int newDim, Initializer wInitializer)
	{
		super(name, newDim);
		this.wInitializer = wInitializer;
	}
	
	@Override
	public void setup()
	{
		super.setup();
		setupW();
	}
	
	protected void setupW()
	{
		this.W = new ParamUnit("W[" + this.name + "]", outDim, input.dim());
		this.wInitializer.init(W);
	}
}
