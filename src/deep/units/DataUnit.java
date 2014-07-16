package deep.units;

import deep.DeepException;
import utils.PP;
import gpu.FloatMat;

public class DataUnit extends Unit
{
    protected FloatMat data = null;
    protected FloatMat gradient = null;
    
    // If the last batch is smaller than the previous ones
    private int batchSize = -1;
    private FloatMat dataSub = null; // offset-mat: doesn't contain any GPU memory
    private FloatMat gradientSub = null; // offset-mat: doesn't contain any GPU memory
    
    /**
     *  Dummy placeholder: if gradient == FloatMat.DUMMY, 
     *  we calculate but don't store the gradient
     */
	public DataUnit(String name, FloatMat data, FloatMat gradient)
	{
		super(name);
		this.data = data;
		this.gradient = gradient;
		if (data != null)
    		this.batchSize = data.col;
		// For offset purpose
		this.dataSub = data;
		this.gradientSub = gradient;
	}

	/**
	 * Ctor for units without gradient
	 */
	public DataUnit(String name, FloatMat data)
	{
		this(name, data, null);
	}
	
	/**
	 * @return data.row
	 */
	public int dim() { return data.row; }

	/**
	 * @return Might be data.col, might be fewer
	 */
	public int batchSize() 
	{	
		if (batchSize < 0)
		{
			if (data != null)
    			this.batchSize = data.col;
			else
				throw new DeepException("Cannot get batchSize: data == null");
		}
		return batchSize;
	}
	
	/**
	 *  If the next batch has fewer columns than the previous
	 *  We createOffset under the hood
	 */
	public void setBatchSize(int newBatchSize)
	{
		this.batchSize = newBatchSize;
		// less than a previous batch in the latest data
		if (batchSize < data.col)
		{
			this.dataSub = this.data.createColOffset(0, batchSize);
			if (isGradientComputed())
				this.gradientSub = this.gradient.createColOffset(0, batchSize);
		}
		else if (batchSize > data.col)
			throw new DeepException("Batch size exceeds GPU 'data' matrix col dim.");
	}
	
	/**
	 * Should use this to access 'data' field
	 */
	public FloatMat data()
	{
		return batchSize < data.col ? this.dataSub : this.data;
	}

	/**
	 * Should use this to access 'data' field
	 */
	public FloatMat gradient()
	{
		return batchSize < data.col ? this.gradientSub : this.gradient;
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
