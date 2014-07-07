package deep.units;

import deep.DeepException;
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
	 * Reinitialize this parameter with parent's initer
	 */
	public void reInit()
	{ 
		if (parent == null)
			throw new DeepException("Cannot reinitialize this parameter: parent null");
		parent.reInit();
	}
}
