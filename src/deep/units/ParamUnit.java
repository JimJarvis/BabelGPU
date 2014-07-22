package deep.units;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import utils.FileUtil;
import deep.*;
import gpu.FloatMat;

public class ParamUnit extends DataUnit
{
	private static final long serialVersionUID = 1L;

	/**
	 * Unless in debug mode, we don't explicitly store the parameter gradient
	 * Instantiate a new FloatMat with (row, col)
	 * @param parent the ParamComputeUnit that uses this parameter
	 */
	public ParamUnit(String name, ParamComputeUnit parent, int row, int col)
	{
		super(name, parent, new FloatMat(row, col));
		setDummyGradient();
	}
	
	/**
	 * No parent ParamComputeUnit: this is not a usual ParamUnit
	 * @see FourierProjectUnit
	 */
	public ParamUnit(String name, int row, int col)
	{
		this(name, null, row, col);
	}
	
	/**
	 * Copy ctor: make a partial copy of a submat of 'data'
	 * @param 
	 */
	
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
	
	/* ***********************
	 * Customized serialization: we only save 'data' field
	 */
	/**
	 * 'data' field stored to disk by serialization
	 */
	public String getDataFilePath()
	{
		return FileUtil.join(this.parent.learningPlan.dir, name + ".float");
	}
	
	private void writeObject(ObjectOutputStream out) throws IOException
	{
		out.defaultWriteObject();
		out.writeObject(data.saveable(getDataFilePath()));
	}
	
	/**
	 * Must have EXACTLY the same signature for JVM to serialize
	 */
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException
	{
		in.defaultReadObject();
		this.data = new FloatMat((FloatMat.Saveable) in.readObject());
	}
}
