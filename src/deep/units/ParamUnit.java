package deep.units;

import gpu.FloatMat;

public class ParamUnit extends DataUnit
{
	/**
	 * Unless in debug mode, we don't explicitly store the parameter gradient
	 */
	public ParamUnit(String name, int row, int col)
	{
		super(name, new FloatMat(row, col), DUMMY);
	}
}
