package deep;

import deep.units.ComputeUnit;
import deep.units.TerminalUnit;

public class DeepNet
{
	private ComputeUnit head;
	private TerminalUnit terminal;
	
	public DeepNet(ComputeUnit ... units)
	{
		this.head = units[0];
		this.terminal = (TerminalUnit) units[units.length - 1];
		chain(units);
	}
	
	public DeepNet(ComputeUnit head, TerminalUnit terminal)
	{
		this.head = head;
		this.terminal = terminal;
	}
	
	public static void chain(ComputeUnit ... units)
	{
		int len = units.length;
		for (int i = 0; i < len; i ++)
		{
			if (i != 0)
				units[i].prev = units[i - 1];
			if (i != len - 1)
				units[i].next = units[i + 1];
		}
	}
	
	public void forwprop()
	{
		ComputeUnit unitptr = this.head;
		while (unitptr != null)
		{
			unitptr.forward();
			unitptr = unitptr.next;
		}
	}

	public void backprop()
	{
		ComputeUnit unitptr = this.terminal;
		while (unitptr != null)
		{
			unitptr.backward();
			unitptr = unitptr.prev;
		}
	}
}