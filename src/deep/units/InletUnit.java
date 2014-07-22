package deep.units;

import com.googlecode.javacpp.IntPointer;
import deep.*;
import gpu.FloatMat;

public abstract class InletUnit extends DataUnit
{
	private static final long serialVersionUID = 1L;
	public transient FloatMat goldMat;
	public transient IntPointer goldLabels;
	public final int MaxBatchSize;
	/**
	 * Critical field: needed across all ComputeUnits in the entire net
	 * updated by every call to nextBatch()
	 */
	public int batchSize = -1; 
	
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
	 * Update the critical 'batchSize' field used across the entire net. 
	 * Update learningPlan.doneSampleSize
	 * @return this.batchSize
	 * @see #nextBatch_() public interface to the abstract method
	 */
	public int nextBatch()
	{
		this.batchSize = this.nextBatch_();
		if (this.batchSize <= 0)
			throw new DeepException("InletUnit should not yield batchSize " + batchSize);
		this.parent.learningPlan.doneSampleSize += this.batchSize;
		return this.batchSize;
	}

	/**
	 * Update 'data' field 
	 * Should be called right after {@link DeepNet#hasNext()} 
	 * @return batch size of this batch, might be less than MaxBatchSize
	 */
	protected abstract int nextBatch_();

	/**
	 * Either load to 'goldMat' or 'goldLabels'
	 */
	public abstract void nextGold();

	/**
	 * Reset the inlet stream to prepare for the next epoch from start
	 */
	public abstract void prepareNextEpoch();
	
	/**
	 * Overridable, defaults to {@link #prepareNextEpoch()}
	 */
	public void reset()
	{
		this.prepareNextEpoch();
	}
}
