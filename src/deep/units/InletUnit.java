package deep.units;

import gpu.FloatMat;

public abstract class InletUnit extends DataUnit
{
	public FloatMat goldMat;
	
	public InletUnit(String name)
	{
		super(name, null);
	}

	public abstract boolean hasNext();
	public abstract void nextBatch();
	public abstract void nextGold();
}