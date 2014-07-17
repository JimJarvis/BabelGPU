package deep.units;

import com.googlecode.javacpp.IntPointer;

import deep.DeepException;
import gpu.FloatMat;

public abstract class InletUnit extends DataUnit
{
	public FloatMat goldMat;
	public IntPointer goldLabels;
	
	/**
	 * Inlet doesn't have gradient
	 * @param data mustn't be null. Include the extra row for bias if necessary
	 */
	public InletUnit(String name, FloatMat data)
	{
		super(name, null, data, null);
		if (data == null)
			throw new DeepException("Inlet data mustn't be null");
	}

	/**
	 * Construct a new 'data' matrix with row == 'dim' and col == 'batchSize'
	 * @param dim raw row dimension (may include extra bias row)
	 * NOTE: you're responsible for adding an extra row for bias here
	 */
	public InletUnit(String name, int dim, int batchSize)
	{
		super(name, null, new FloatMat(dim, batchSize), null);
	}
	
	/**
	 * Link this InletUnit to a parent ComputeUnit
	 * Should be done before setup
	 */
	public void setParent(ComputeUnit parent)
	{
		this.parent = parent;
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
