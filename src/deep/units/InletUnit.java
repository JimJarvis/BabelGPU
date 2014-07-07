package deep.units;

import gpu.FloatMat;

public abstract class InletUnit extends DataUnit
{
	public FloatMat goldMat;
	
	public InletUnit(String name)
	{
		super(name, null, null);
		setNoGradient(); // conceptual
	}

	public InletUnit(String name, int row, int col)
	{
		super(name, new FloatMat(row, col), null);
		setNoGradient();
	}

	public abstract boolean hasNext();
	public abstract void nextBatch();
	public abstract void nextGold();
	public abstract void reset();
}
