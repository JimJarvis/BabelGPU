package deep.units;

import utils.PP;
import gpu.FloatMat;

public class DataUnit extends Unit
{
    public FloatMat data = null;
    public FloatMat gradient = null;
    // Dummy value: if gradient == DUMMY, we calculate but don't store the gradient
    public static final FloatMat DUMMY = new FloatMat();
    
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
	
	public void setDummyGradient()
	{
		this.gradient = DUMMY;
	}
	
	public void setNoGradient()
	{
		this.gradient = null;
	}
	
	// Debug only
	public String toString()
	{
		return PP.all2str("[", name, "]\n<Data>", data, "\n<Gradient>", gradient);
	}
}
