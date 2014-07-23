package deep.units;

import java.io.*;
import deep.*;
import gpu.FloatMat;

public class ParamUnit extends DataUnit
{
	private static final long serialVersionUID = 1L;

	/**
	 * Unless in debug mode, we don't explicitly store the parameter gradient
	 * Instantiate a new FloatMat with (row, col)
	 * saveMode defaults to SAVE_GRAD: serialize 'data' to disk
	 * @param parent the ParamComputeUnit that uses this parameter
	 */
	public ParamUnit(String name, ComputeUnit parent, int row, int col)
	{
		super(name, parent, new FloatMat(row, col));
		setDummyGradient();
		this.saveMode = DataUnit.SAVE_DATA;
	}
	
	@Override
	public final int dim()
	{
		throw new DeepException("ParamUnit doesn't have 'dim'. Use data.row instead");
	}
	
	/**
	 * Simply 'data'. No colOffset
	 */
	@Override
	public FloatMat data() { return this.data; }

	/**
	 * Simply 'gradient'. No colOffset
	 */
	@Override
	public FloatMat gradient() { return this.gradient; }
	
	/**
	 * Copies ONLY the FloatMat 'data'
	 */
	public void copyDataFrom(ParamUnit other)
	{
		if (this.data.row != other.data.row || this.data.col != other.data.col)
			throw new DeepException("Cannot copy data from a different dimension.");
		this.data.copyFrom(other.data);
	}
	
	/**
	 * Reinitialize this parameter with parent's initer
	 */
	public void reInit()
	{ 
		if (parent == null)
			throw new DeepException("Cannot reinitialize this parameter: parent null");
		((ParamComputeUnit) parent).reInit();
	}
	
	// Internal serialization: copy this for subclasses
	private void writeObject(ObjectOutputStream out) throws IOException
	{
		this.serialize(out);
	}
	
	// Internal serialization: copy this for subclasses
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
	{
		this.deserialize(in);
	}
}
