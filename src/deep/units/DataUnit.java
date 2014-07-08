package deep.units;

import utils.PP;
import gpu.FloatMat;

public class DataUnit extends Unit
{
    public FloatMat data = null;
    public FloatMat gradient = null;
    
    /**
     *  Dummy placeholder: if gradient == FloatMat.DUMMY, 
     *  we calculate but don't store the gradient
     */
	public DataUnit(String name, FloatMat data, FloatMat gradient)
	{
		super(name);
		this.data = data;
		this.gradient = gradient;
	}

	/**
	 * Ctor for units without gradient
	 */
	public DataUnit(String name, FloatMat data)
	{
		this(name, data, null);
	}
	
	public int dim() { return data.row; }
	
	public int batchSize() {	return data.col;	}
	
	public void initGradient()
	{
		this.gradient = new FloatMat(this.data);
	}
    
	public boolean hasGradient()
	{
		return this.gradient != null;
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
