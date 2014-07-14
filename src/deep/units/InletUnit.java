package deep.units;

import com.googlecode.javacpp.IntPointer;
import gpu.FloatMat;

public abstract class InletUnit extends DataUnit
{
	public FloatMat goldMat;
	public IntPointer goldLabels;
	
	/**
	 * Inlet doesn't have gradient
	 */
	public InletUnit(String name)
	{
		super(name, null, null);
	}

	public InletUnit(String name, int row, int col)
	{
		super(name, new FloatMat(row, col), null);
	}

	public abstract boolean hasNext();
	public abstract void nextBatch();
	public abstract void nextGold();
	public abstract void reset();
}
