package deep.units;

public abstract class Unit
{
	public String name;
	protected boolean debug = false;
	
	public Unit(String name)
	{
		this.name = name;
	}
	
	public void enableDebug(boolean debug)
	{
		this.debug = debug;
	}
}
