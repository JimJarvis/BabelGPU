package deep.units;

import deep.DeepException;
import gpu.FloatMat;

public class ParamUnit extends DataUnit
{
	/**
	 * Unless in debug mode, we don't explicitly store the parameter gradient
	 */
	public ParamUnit(String name, int row, int col)
	{
		super(name, new FloatMat(row, col));
		setDummyGradient();
	}
	
	@Override
	public final int dim()
	{
		throw new DeepException("ParamUnit doesn't have 'dim'. Use row() instead");
	}
	
	@Override
	public final int batchSize()
	{
		throw new DeepException("ParamUnit doesn't have 'batchSize', use col() instead");
	}
	
	public int row() {	return this.data.row;	}
	public int col() {	return this.data.col;	}
}
