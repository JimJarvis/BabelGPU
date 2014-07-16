package deep.units;

import deep.*;
import gpu.FloatMat;

public class ParamUnit extends DataUnit
{
	private ParamComputeUnit parent;
	
	/**
	 * Unless in debug mode, we don't explicitly store the parameter gradient
	 * @param parent the ParamComputeUnit that uses this parameter
	 */
	public ParamUnit(String name, ParamComputeUnit parent, int row, int col)
	{
		super(name, new FloatMat(row, col));
		setDummyGradient();
		this.parent = parent;
	}
	
	/**
	 * Infer row/col dimensions from 'parent' ParamComputeUnit
	 * rowDim = parent.outDim
	 * colDim = parent.input.dim
	 */
	public ParamUnit(String name, ParamComputeUnit parent)
	{
		this(name, parent, parent.outDim, parent.input.dim());
	}
	
	/**
	 * No parent ParamComputeUnit: this is not a usual ParamUnit
	 * @see FourierProjectUnit
	 */
	public ParamUnit(String name, int row, int col)
	{
		this(name, null, row, col);
	}
	
	/**
	 * Shallow copy
	 */
	public ParamUnit(ParamUnit other)
	{
		super(other.name, other.data, other.gradient);
		this.debug = other.debug;
		this.parent = other.parent;
	}
	
	/**
	 * Copy ctor: make a partial copy of a submat of 'data'
	 * @param 
	 */
	
	@Override
	public final int dim()
	{
		throw new DeepException("ParamUnit doesn't have 'dim'. Use data.row instead");
	}
	
	@Override
	public final int batchSize()
	{
		throw new DeepException("ParamUnit doesn't have 'batchSize', use data.col instead");
	}
	
	/**
	 * Simply 'data'. No colOffset
	 */
	@Override
	public FloatMat data() { return this.data; }

	/**
	 * Simply 'gradient'. No colOffset
	 */
	@Override
	public FloatMat gradient() { return this.gradient; }
	
	/**
	 * Reinitialize this parameter with parent's initer
	 */
	public void reInit()
	{ 
		if (parent == null)
			throw new DeepException("Cannot reinitialize this parameter: parent null");
		parent.reInit();
	}
}
