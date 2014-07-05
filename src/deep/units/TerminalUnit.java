package deep.units;

public abstract class TerminalUnit extends PureComputeUnit
{
	public DataUnit y;
	protected float result = 0;

	public TerminalUnit(String name, DataUnit y)
	{
		super(name);
		this.y = y;
	}
	
	public float getResult() {	return result;	}
	
	public float update(float update) {	return result += update;	}
}
