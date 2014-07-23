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
	protected void setupW_()
	{
		this.initer.setBias(hasBias);
		this.W = new ParamUnit(
				"Param#" + this.name,
				this,
				this.outDim, 
				this.input.dim());
		reInit();
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
    			getPlan().regScheme.regGradUpdate(this);
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
    			getPlan().regScheme.regParamUpdate(this);
        		GpuBlas.mult(output.gradient(), input.data().transpose(), W.data(), -lr, 1);
    		}
    		if (hasBias)
    			W.data().fillLastRow0();
		}
	}

}
