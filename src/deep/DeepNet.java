package deep;

import java.util.Iterator;

import deep.units.ComputeUnit;
import deep.units.TerminalUnit;

public class DeepNet implements Iterable<ComputeUnit>
{
	private ComputeUnit head;
	private TerminalUnit terminal;

	public DeepNet(ComputeUnit... units)
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

	public static void chain(ComputeUnit... units)
	{
		int len = units.length;
		for (int i = 0; i < len; i++)
		{
			if (i != 0) units[i].prev = units[i - 1];
			if (i != len - 1) units[i].next = units[i + 1];
		}
	}

	public void setup()
	{
		for (ComputeUnit unit : this)
			unit.setup();
	}

	public void forwprop()
	{
		for (ComputeUnit unit : this)
			unit.forward();
	}

	public void backprop()
	{
		for (ComputeUnit unit : iterable(false))
			unit.backward();
	}

	public void run()
	{

	}

	// ******************** Enable forward/backward iteration ********************/
	public Iterable<ComputeUnit> iterable(final boolean forward)
	{
		return new Iterable<ComputeUnit>()
		{
			public Iterator<ComputeUnit> iterator()
			{
				return DeepNet.this.iterator(forward);
			}
		};
	}

	@Override
	public Iterator<ComputeUnit> iterator()
	{
		return iterator(true);
	}

	public Iterator<ComputeUnit> iterator(final boolean forward)
	{
		return new Iterator<ComputeUnit>()
		{
			ComputeUnit unitptr;
			{
				unitptr = forward ? 
						DeepNet.this.head : DeepNet.this.terminal;
			}

			@Override
			public boolean hasNext() { return unitptr != null; }

			ComputeUnit tmpptr;
			@Override
			public ComputeUnit next()
			{
				tmpptr = unitptr;
				unitptr = forward ? unitptr.next : unitptr.prev;
				return tmpptr;
			}

			public void remove() { }
		};
	}
}