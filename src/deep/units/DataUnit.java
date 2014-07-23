package deep.units;

import java.io.*;
import deep.*;
import utils.*;
import gpu.FloatMat;

public class DataUnit extends Unit
{
    private static final long serialVersionUID = 1L;
	protected transient FloatMat data;
    protected transient FloatMat gradient;
    public ComputeUnit parent;
    
   /**
    * Serialization
    * FloatMats can't be saved directly, so we actually save the underlying 
    * float[] to disk. You can choose what to save: 'data', 'gradient' or both by 
    * the OR-operator SAVE_DATA | SAVE_GRADIENT 
    * @see FloatMat#saveable(String)
    */
    public int saveMode = 0;
    public static final int SAVE_DATA = 1;
    public static final int SAVE_GRADIENT = 2;
    
    /**
     *  Dummy placeholder: if gradient == FloatMat.DUMMY, 
     *  we calculate but don't store the gradient
     */
	public DataUnit(String name, ComputeUnit parent, FloatMat data, FloatMat gradient)
	{
		super(name);
		this.data = data;
		this.gradient = gradient;
		this.parent = parent;
	}

	/**
	 * Ctor for units without gradient
	 */
	public DataUnit(String name, ComputeUnit parent, FloatMat data)
	{
		this(name, parent, data, null);
	}
	
	/**
	 * @return data.row
	 */
	public int dim() { return data.row; }
	
	/**
	 * Should use this to access 'data' field
	 */
	public FloatMat data()
	{
		int batchSize = this.parent.inlet.batchSize;
		return data != null && batchSize < data.col ? 
				this.data.createColOffset(0, batchSize) : this.data;
	}

	/**
	 * Should use this to access 'data' field
	 */
	public FloatMat gradient()
	{
		int batchSize = this.parent.inlet.batchSize;
		return gradient != null && batchSize  < data.col ? 
				this.gradient.createColOffset(0, batchSize) : this.gradient;
	}
	
	public void initGradient()
	{
		this.gradient = new FloatMat(this.data);
	}
    
	public boolean hasGradient()
	{
		return this.gradient != null;
	}
	
	/**
	 * Sometimes we don't explicitly compute the gradient, but do SGD on W.data directly
	 */
	public boolean isGradientComputed()
	{
		return hasGradient() && !FloatMat.isDummy(gradient);
	}
	
	/**
	 * Carefully release FloatMat resource if already allocated
	 */
	public void setDummyGradient()
	{
		FloatMat.destroy(this.gradient);
		this.gradient = FloatMat.DUMMY;
	}
	
	/**
	 * Carefully release FloatMat resource if already allocated
	 */
	public void setNoGradient()
	{
		FloatMat.destroy(this.gradient);
		this.gradient = null;
	}
	
	/**
	 * @return learningPlan from parent
	 */
	@Override
	public LearningPlan getPlan()
	{
		if (parent == null)
			throw new DeepException("Not associated to a parent ComputeUnit");
		return parent.getPlan();
	}
	
	// ******************** Serialization ********************/
	/**
	 * SAVE_DATA or
	 * SAVE_GRADIENT or 
	 * SAVE_DATA | SAVE_GRADIENT for both
	 */
	public void setSaveMode(int saveMode)
	{
		this.saveMode = saveMode;
	}
	/**
	 * Are we saving 'data'?
	 */
	public boolean doesSaveData()
	{
		return (this.saveMode & SAVE_DATA) != 0;
	}
	/**
	 * Are we saving 'gradient'?
	 */
	public boolean doesSaveGradient()
	{
		return (this.saveMode & SAVE_GRADIENT) != 0;
	}
	
	/**
	 * Save to object stream 'data' and 'gradient', as specified by saveMode 
	 * Used in internal serialization mechanism. 
	 */
	public void serialize(ObjectOutputStream out) throws IOException
	{
		out.defaultWriteObject(); // must have
		if (doesSaveData())
		{
    		String filePath = FileUtil.join(this.getPlan().dir, name + ".float");
    		if (data == null)
    			throw new DeepException("data field is null: cannot serialize");
    		out.writeObject(FloatMat.saveable(data, filePath));
		}
		if (doesSaveGradient())
		{
    		String filePath = FileUtil.join(this.getPlan().dir, name + "_gradient.float");
    		if (gradient == null)
    			throw new DeepException("gradient field is null: cannot serialize");
    		out.writeObject(FloatMat.saveable(gradient, filePath));
		}
	}
	
	/**
	 * Restore from object stream 'data' and 'gradient', as specified by saveMode
	 * Used in internal deserialization mechanism
	 */
	public void deserialize(ObjectInputStream in) throws ClassNotFoundException, IOException
	{
		in.defaultReadObject();
		if (doesSaveData())
			this.data = FloatMat.desaveable((FloatMat.Saveable) in.readObject());
		if (doesSaveGradient())
			this.gradient = FloatMat.desaveable((FloatMat.Saveable) in.readObject());
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
	
	// Debug only
	public String toString()
	{
		return PP.all2str("\"" + name + "\"\n<Data>", data, "\n<Gradient>", gradient);
	}
}