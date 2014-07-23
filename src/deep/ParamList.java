package deep;

import java.util.ArrayList;

import deep.units.*;

public class ParamList extends ArrayList<ParamUnit>
{
	private static final long serialVersionUID = 1L;

	public ParamList()
	{
		super();
	}
	
	/**
	 * Construct a parameter list from a DeepNet in forward order
	 */
	public ParamList(DeepNet net)
	{
		super();
		for (ComputeUnit unit : net)
			if (unit instanceof ParamComputeUnit)
				this.add(((ParamComputeUnit) unit).W);
	}
	
	/**
	 * Construct from an existing param list: deep copying data
	 * SaveMode will be the same as the old one. 
	 * @param nameSuffix name the new ParamUnits based on the old
	 */
	public ParamList(ParamList other, String nameSuffix)
	{
		super();
		for (ParamUnit W : other)
		{
			ParamUnit newW = 
					new ParamUnit(W.name + nameSuffix, 
							(ParamComputeUnit) W.parent, 
							W.data().row, W.data().col);
			newW.copyDataFrom(W);
			newW.setSaveMode(W.saveMode);
			this.add(newW);
		}
	}
	
	/**
	 * Deep copy data from another list
	 */
	public void copyDataFrom(ParamList other)
	{
		for (int i = 0; i < size(); i++)
		{
			ParamUnit thisW = this.get(i);
			ParamUnit otherW = other.get(i);
			if (thisW == null || otherW == null)
				throw new DeepException("None of ParamList elements can be null. ");
			thisW.copyDataFrom(otherW);
		}
	}
}
