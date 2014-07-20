package deep.units;

import deep.Initializer;
import gpu.GpuBlas;

public class LinearUnit extends ParamComputeUnit
{
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
		this.initer.setBias(hasBias);
		this.W = new ParamUnit(
				"Param#" + this.name,
				this,
				this.outDim, 
				this.input.dim());
		reInit();
		if (debug)
			this.W.initGradient();
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
    		// update W with reg
    		float lr = learningPlan.lrStart;
    		// In debug mode, we explicitly store the parameter gradient
    		if (debug)
    		{
    			GpuBlas.mult(output.gradient(), input.data().transpose(), W.gradient());
    			GpuBlas.scaleAdd(W.data(), W.gradient(), learningPlan.reg);
    			if (hasBias) W.gradient().fillLastRow0();
    		}

    		// division by batchSize should be done in the terminal unit
    		GpuBlas.mult(output.gradient(), input.data().transpose(), W.data(), 
                				- lr, 1 - lr * learningPlan.reg);
    		if (hasBias)
    			W.data().fillLastRow0();
		}
	}

}
