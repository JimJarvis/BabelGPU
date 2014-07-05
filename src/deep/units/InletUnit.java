package deep.units;

public abstract class InletUnit extends DataUnit
{

	public InletUnit(String name)
	{
		super(name, null);
	}

	public abstract boolean nextBatch();
}
