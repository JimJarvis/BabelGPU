package deep.units;

import java.io.Serializable;
public abstract class Unit implements Serializable
{
	private static final long serialVersionUID = 1L;

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