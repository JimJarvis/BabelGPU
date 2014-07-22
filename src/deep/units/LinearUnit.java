package deep.units;

import deep.*;
import deep.RegScheme.L2RegScheme;
import gpu.GpuBlas;

public class LinearUnit extends ParamComputeUnit
{
	private static final long serialVersionUID = 1L;

	public LinearUnit(String name, InletUnit inlet, int outDim, boolean hasBias, Initializer initer)
	{
		super(name, inlet, outDim, hasBias, initer);
	}

	public LinearUnit(String name, InletUnit inlet, int outDim, Initializer wInitializer)
	{
		super(name, inlet, outDim, wInitializer);
	}
	
	@Override
	protected void setupW()
	{
		// W isn't null if we load from serialized data on disk. 
		if (W == null)
		{
    		this.initer.setBias(hasBias);
    		this.W = new ParamUnit(
    				"Param#" + this.name,
    				this,
    				this.outDim, 
    				this.input.dim());
    		reInit();
		}
		if (debug)
			this.W.initGradient();
		else
			this.W.setDummyGradient();
	}
	
	@Override
	public void forward()
	{
		if (hasBias)
			input.data().fillLastRow1();
		GpuBlas.mult(W.data(), input.data(), output.data());
	}

	@Override
	public void backward()
	{
		// update input.gradient() only when necessary 
		// Don't upgrade the gradient of the inlet layer, of course
		if (input.hasGradient())
			GpuBlas.mult(W.data().transpose(), output.gradient(), input.gradient());

		if (W.hasGradient())
		{
    		// In debug mode, we explicitly store the parameter gradient
    		if (debug)
    		{
    			GpuBlas.mult(output.gradient(), input.data().transpose(), W.gradient());
    			GpuBlas.scaleAdd(W.data(), W.gradient(), getPlan().reg);
    			if (hasBias) W.gradient().fillLastRow0();
    		}

    		// update W with reg
    		float lr = getPlan().lr;
    		
    		// Optimization specific to L2 regularizer
    		// division by batchSize should be done in the terminal unit
    		if (getPlan().regScheme instanceof L2RegScheme)
        		GpuBlas.mult(output.gradient(), input.data().transpose(), W.data(), 
                    				- lr, 1 - lr * getPlan().reg);
    		else
    		{
    			getPlan().regScheme.regGradUpdate(this);
        		GpuBlas.mult(output.gradient(), input.data().transpose(), W.data(), -lr, 1);
    		}
    		if (hasBias)
    			W.data().fillLastRow0();
		}
	}

}
