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
	
	public int size() {	return this.data.size();	}
	
	/**
	 * Reinitialize this parameter with parent's wInitializer
	 */
	public void reInit() { parent.reInit(); }
}
