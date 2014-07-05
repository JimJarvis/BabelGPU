package deep.units;

import gpu.FloatMat;

public class ParamUnit extends DataUnit
{
	public ParamUnit(String name, int row, int col)
	{
		super(name, new FloatMat(row, col), DUMMY);
	}
}
