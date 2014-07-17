package deep.units;

import utils.PP;
import gpu.FloatMat;

public class DataUnit extends Unit
{
    protected FloatMat data;
    protected FloatMat gradient;
    protected ComputeUnit parent;
    
    /**
     *  Dummy placeholder: if gradient == FloatMat.DUMMY, 
     *  we calculate but don't store the gradient
     */
	public DataUnit(String name, ComputeUnit parent, FloatMat data, FloatMat gradient)
	{
		super(name);
		this.data = data;
		this.gradient = gradient;
		this.parent = parent;
	}

	/**
	 * Ctor for units without gradient
	 */
	public DataUnit(String name, ComputeUnit parent, FloatMat data)
	{
		this(name, parent, data, null);
	}
	
	/**
	 * @return data.row
	 */
	public int dim() { return data.row; }
	
	/**
	 * Should use this to access 'data' field
	 */
	public FloatMat data()
	{
		int batchSize = this.parent.inlet.batchSize();
		return data != null && batchSize < data.col ? 
				this.data.createColOffset(0, batchSize) : this.data;
	}

	/**
	 * Should use this to access 'data' field
	 */
	public FloatMat gradient()
	{
		int batchSize = this.parent.inlet.batchSize();
		return gradient != null && batchSize  < data.col ? 
				this.gradient.createColOffset(0, batchSize) : this.gradient;
	}
	
	public void initGradient()
	{
		this.gradient = new FloatMat(this.data);
	}
    
	public boolean hasGradient()
	{
		return this.gradient != null;
	}
	
	/**
	 * Sometimes we don't explicitly compute the gradient, but do SGD on W.data directly
	 */
	public boolean isGradientComputed()
	{
		return hasGradient() && this.gradient != FloatMat.DUMMY;
	}
	
	/**
	 * Carefully release FloatMat resource if already allocated
	 */
	public void setDummyGradient()
	{
		FloatMat.destroy(this.gradient);
		this.gradient = FloatMat.DUMMY;
	}
	
	/**
	 * Carefully release FloatMat resource if already allocated
	 */
	public void setNoGradient()
	{
		FloatMat.destroy(this.gradient);
		this.gradient = null;
	}
	
	// Debug only
	public String toString()
	{
		return PP.all2str("\"" + name + "\"\n<Data>", data, "\n<Gradient>", gradient);
	}
}
