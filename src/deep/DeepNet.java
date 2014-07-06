package deep;

import java.util.*;

import utils.PP;
import deep.units.*;

public class DeepNet implements Iterable<ComputeUnit>
{
	private ComputeUnit head;
	private InletUnit inlet;
	private TerminalUnit terminal;

	public DeepNet(ComputeUnit... units)
	{
		this.head = units[0];
		this.terminal = (TerminalUnit) units[units.length - 1];
		this.inlet = (InletUnit) head.input;
		chain(units);
	}
	
	public DeepNet(ArrayList<ComputeUnit> units)
	{
		this(units.toArray(new ComputeUnit[units.size()]));
	}

	/**
	 * If the network between head and terminal is already chained
	 */
	public DeepNet(ComputeUnit head, TerminalUnit terminal)
	{
		this.head = head;
		this.terminal = terminal;
		this.inlet = (InletUnit) head.input;
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
	
	public void setLearningPlan(LearningPlan learningPlan)
	{
		for (ComputeUnit unit : this)
			unit.learningPlan = learningPlan;
	}
	
	/**
	 * If debug mode enabled, we explicitly store the parameter gradient
	 */
	public void enableDebug(boolean debug)
	{
		for (ComputeUnit unit : this)
			unit.enableDebug(debug);
	}
	public void enableDebug() {	this.enableDebug(true); }

	public void run(LearningPlan learningPlan)
	{
		setLearningPlan(learningPlan);
		setup();
		
		while (inlet.hasNext())
		{
			inlet.nextBatch();
			forwprop();
			backprop();
		}
	}
	
	/**
	 * Fill all compute units with default generated name
	 * @return this
	 */
	public DeepNet genDefaultUnitName()
	{
		HashMap<String, Integer> map = new HashMap<>();
		String className;
		for (ComputeUnit unit : this)
		{
			className = unit.getClass().getSimpleName();
			// className always ends with 'Unit'
			className = className.substring(0, className.length() - 4);
			Integer idx = map.get(className);
			if (idx == null)
			{
				map.put(className, 1);
				idx = 1;
			}
			else // use the last index + 1
				map.put(className, ++ idx);
			unit.name = String.format("%s{%d}", className, idx);
		}
		return this;
	}
	
	/**
	 * @return a new HashMap that maps name to ComputeUnit
	 */
	public HashMap<String, ComputeUnit> getUnitMap()
	{
		HashMap<String, ComputeUnit> unitMap = new HashMap<>();
		for (ComputeUnit unit : this)
			unitMap.put(unit.name, unit);
		return unitMap;
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
	
	// ******************** DEBUG only ********************/
	public void runDebug(LearningPlan learningPlan)
	{
	 	setLearningPlan(learningPlan);
	 	enableDebug();
	 	PP.pTitledSectionLine("SETUP");
		setup();
		printDebug();
		int i = 1;
		
		while (inlet.hasNext())
		{
			PP.pSectionLine("=", 90);
			PP.p("Iteration", i++, "reading inlet");
			inlet.nextBatch();
			PP.pTitledSectionLine("FORWARD");
			forwprop();
			printDebug();
			PP.pTitledSectionLine("BACKWARD");
			backprop();
			printDebug();
		}
		
		PP.p("\nRESULT =", terminal.getResult());
	}
	
	public void printDebug()
	{
		for (ComputeUnit unit : this)
		{
			PP.p(unit.name);
			PP.p("input:", unit.input);
			if (unit instanceof ParamComputeUnit)
				PP.p("W:", ((ParamComputeUnit) unit).W);
			PP.pSectionLine();
		}
	}
}