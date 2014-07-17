package deep.units;

import com.googlecode.javacpp.IntPointer;

import deep.DeepException;
import deep.DeepNet;
import gpu.FloatMat;

public abstract class InletUnit extends DataUnit
{
	public FloatMat goldMat;
	public IntPointer goldLabels;
	public final int MaxBatchSize;
	
	/**
	 * Inlet doesn't have gradient
	 * maxBatchSize == data.col
	 * @param data mustn't be null. Include the extra row for bias if necessary
	 */
	public InletUnit(String name, FloatMat data)
	{
		super(name, null, data, null);
		if (data == null)
			throw new DeepException("Inlet data mustn't be null");
		this.MaxBatchSize = data.col;
	}

	/**
	 * Construct a new 'data' matrix with row == 'dim' and col == 'batchSize'
	 * @param dim raw row dimension (may include extra bias row)
	 * @param MaxBatchSize for initiating (gpu alloc) 'output' in the Net.
	 * NOTE: you're responsible for adding an extra row for bias here
	 * @param actuallyAlloc true to actually allocate memory for 'data'. 
	 * false to fill 'data' with a dummy FloatMat that only has row/col dim info
	 * @see FloatMat#createDummyMat(int, int)
	 */
	public InletUnit(String name, int dim, int MaxBatchSize, boolean actuallyAlloc)
	{
		super(name, null, 
				actuallyAlloc ? 
						new FloatMat(dim, MaxBatchSize) :
						FloatMat.createDummyMat(dim, MaxBatchSize), 
				null);
		this.MaxBatchSize = MaxBatchSize;
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
	 * Will be called every time AFTER each 'nextBatch' to determine 
	 * how many cols actually need to be used. 
	 */
	public abstract int batchSize();

	/**
	 * Update 'data' field 
	 * Should be called right after {@link DeepNet#hasNext()}
	 * @see #batchSize() set batchSize right after
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
