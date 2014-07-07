package deep;

import gpu.FloatMat;

import java.util.*;

import utils.PP;
import deep.units.*;

public class DeepNet implements Iterable<ComputeUnit>
{
	private ComputeUnit head;
	private InletUnit inlet;
	private TerminalUnit terminal;
	
	private boolean setup = false; // should only setup once

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

	/**
	 * This function call will only setup once and do nothing later
	 */
	public void setup()
	{
		if (!setup)
		{
			for (ComputeUnit unit : this)
    			unit.setup();
			setup = true;
		}
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
	 * Set one initializer for all ParamComputeUnit 
	 */
	public void setInitializer(Initializer initer)
	{
		for (ComputeUnit unit : this)
			if (unit instanceof ParamComputeUnit)
				((ParamComputeUnit) unit).initer = initer;
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
	 * Prepare a network for re-run
	 */
	public void reset()
	{
		Initializer.resetRand();
		for (ParamUnit w : terminal.getParams())
			w.reInit();
		terminal.clearResult();
		terminal.learningPlan.reset();
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
	
	/**
	 * Gradient checking debug routine
	 * Will only load one nextBatch() from inlet
	 */
	public void gradCheck(LearningPlan learningPlan)
	{
		final float EPS = 1e-4f;
		
	 	setLearningPlan(learningPlan);
	 	enableDebug();
		if (!this.setup) // should only read inlet once if we're debugging
    	{
			setup();
			inlet.nextBatch();
    	}
		
		ArrayList<ParamUnit> params = terminal.getParams();
		FloatMat propGrad[] = new FloatMat[params.size()];
		FloatMat goldGrad[] = new FloatMat[params.size()];

		// Get the exact gradient by backprop first
		forwprop();
		backprop();
		
		FloatMat mat;
		
		int i = 0;
		for (ParamUnit w : params)
		{
			mat = new FloatMat(w.gradient);
			mat.copyFrom(w.gradient);
			propGrad[i ++] = mat;
		}
		
		// Do finite-diff forward prop for every entry in every parameter
		i = 0;
		for (ParamUnit w : params)
		{
			mat = new FloatMat(w.data);
			for (int idx = 0 ; idx < w.size(); idx ++)
			{
				// +EPS and -EPS perturb
				float negResult = 0, posResult = 0;
				for (int perturb : new int[] {-1, 1})
				{
    				// Re-init everything as the exact gradient initialization
					this.reset();
    				
            		// Perturb -EPS
            		w.data.singleIncr(idx, perturb * EPS);
            		forwprop();
            		float result = terminal.getResult();

            		if (perturb < 0) negResult = result; else posResult = result;
				}
				// Compute symmetric numerical gradient and store to 'mat'
				mat.singleSet(idx, (posResult - negResult) / (2 * EPS));
			}
			// Store
			goldGrad[i ++] = mat;
		}
		
		PP.setSep("\n\n");
		PP.pTitledSectionLine("BACK-PROP");
		PP.p(propGrad);
		PP.p();
		PP.pTitledSectionLine("Numerical GOLD");
		PP.p(goldGrad);
	}
}