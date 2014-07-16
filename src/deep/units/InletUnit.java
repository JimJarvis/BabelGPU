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

	/**
	 * Update 'data' field
	 * NOTE: 'data' must not have useless columns.
	 * If the last batch has size less than previous batches, use {@link FloatMat#createColOffset}. 
	 * Each column will be updated to LearningPlan as batchSize
	 */
	public abstract void nextBatch();

	/**
	 * Either load to 'goldMat' or 'goldLabels'
	 */
	public abstract void nextGold();

	/**
	 * Reset the inlet stream to prepare for the next epoch from start
	 */
	public abstract void reset();
}
