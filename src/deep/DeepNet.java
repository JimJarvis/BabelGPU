package deep;

import gpu.FloatMat;
import gpu.GpuBlas;

import java.util.*;

import utils.PP;
import deep.units.*;

public class DeepNet implements Iterable<ComputeUnit>
{
	public String name;
	public ComputeUnit head;
	public InletUnit inlet;
	public TerminalUnit terminal;
	
	private boolean setup = false; // should only setup once
	private boolean debug = false; // the whole net is in debug mode

	public DeepNet(String name, InletUnit inlet, ComputeUnit... units)
	{
		this.name = name;
		this.head = units[0];
		this.terminal = (TerminalUnit) units[units.length - 1];
		this.inlet = inlet;
		head.input = inlet;
		chain(units);
	}
	
	public DeepNet(String name, InletUnit inlet, ArrayList<ComputeUnit> units)
	{
		this(name, inlet, units.toArray(new ComputeUnit[units.size()]));
	}

	/**
	 * If the network between head and terminal is already chained
	 */
	public DeepNet(String name, ComputeUnit head, TerminalUnit terminal)
	{
		this.name = name;
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
	 * Any non-ParamComputeUnit before the first ParamComputeUnit doesn't need to calculate gradient
	 * We explicitly disable it.
	 */
	public void setup(LearningPlan learningPlan)
	{
		if (!setup)
		{
			for (ComputeUnit unit : this)
    			unit.setup();
			
			// Explicitly disable gradient calculation in the first few non-paramComputeUnit
			if (!debug)
    			for (ComputeUnit unit : this)
    			{
    				if (unit instanceof ParamComputeUnit)
    					break;
    				else
    					unit.input.setNoGradient();
    			}
			
			setLearningPlan(learningPlan);
			
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
	 * Must be called BEFORE setup() !!
	 */
	public void setBias(boolean hasBias)
	{
		for (ComputeUnit unit : this)
			unit.setBias(hasBias);
	}
	
	/**
	 * If debug mode enabled, we explicitly store the parameter gradient
	 */
	public void enableDebug(boolean debug)
	{
		this.debug = debug;
		for (ComputeUnit unit : this)
			unit.enableDebug(debug);
	}
	public void enableDebug() {	this.enableDebug(true); }

	public void run(LearningPlan learningPlan)
	{
		setup(learningPlan);
		
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
		terminal.clearLoss();
		terminal.learningPlan.reset();
		inlet.reset();
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
	
	/**
	 * @return param list in forward order
	 */
	public ArrayList<ParamUnit> getParams()
	{
		return this.terminal.getParams();
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
	public void runDebug(LearningPlan learningPlan, boolean hasBias)
	{
	 	this.enableDebug();
	 	this.setBias(hasBias); // all have bias units
	 	
		PP.pTitledSectionLine("RUN DEBUG", "=", 25);
	 	PP.pTitledSectionLine("SETUP");
	 	this.setup(learningPlan);
	 	this.printDebug(true);
		
		// Handle debug networks that have no params
		if (terminal.getParams().size() == 0)
			inlet.initGradient();

		int i = 1;
		while (inlet.hasNext())
		{
			PP.pSectionLine("-", 70);
			PP.p("Iteration", i++, "reading inlet");
			inlet.nextBatch();
			PP.pTitledSectionLine("FORWARD");
			forwprop();
			printDebug(true);
			PP.pTitledSectionLine("BACKWARD");
			backprop();
			printDebug(false);
		}
		
		PP.p("\nRESULT =", terminal.lossTotal(), "\n");
	}
	
	public void printDebug(boolean forward)
	{
		for (ComputeUnit unit : this.iterable(forward))
		{
			PP.p(unit.name);
			PP.p("input", unit.input.data().row, "*", unit.input.data().col, ":", unit.input, "\n");
			if (unit instanceof ParamComputeUnit)
			{
				ParamUnit W = ((ParamComputeUnit) unit).W;
				PP.p("W", W.data().row, "*", W.data().col, ":", W);
			}
			PP.pSectionLine();
		}
	}
	
	/**
	 * Gradient checking debug routine
	 * Supports both networks with and without parameters (e.g. pure compute layers)
	 * Will only process one batch from inlet
	 * Perturbation eps constant is auto-selected based on avg abs value of parameters (minimum: 1e-4f)
	 * @param hasBias enable bias units on all ComputeUnit
	 * @param perturbRatio default = 1e3f. perturb EPS = average_abs_val / perturbRatio
	 * @param verbose don't show the actual gradient comparison
	 * @return average percentage error (already multiplied by 100)
	 */
	public float gradCheck(LearningPlan learningPlan, boolean hasBias, float perturbRatio, boolean verbose)
	{
		PP.pTitledSectionLine("GRAD CHECK: " + this.name, "=", 10);
		
	 	this.enableDebug();
	 	this.setBias(hasBias); // all have bias units
	 	
		this.setup(learningPlan);
		this.reset(); inlet.nextBatch();
		
		ArrayList<ParamUnit> params = (ArrayList<ParamUnit>) terminal.getParams().clone();
		// We also do gradient checking for pure computing networks that have no parameters
		boolean hasParams = params.size() != 0;
		
		int gradN = hasParams ? params.size() : 1;
		FloatMat propGrad[] = new FloatMat[gradN];
		FloatMat goldGrad[] = new FloatMat[gradN];
		
		if (!hasParams)
		{
			// add sentinel value: hack the loops below
			params.add(null); 
            // we treat inlet as 'parameter' and compute its finite-difference grad
			inlet.initGradient(); 
		}

		// Get the exact gradient by backprop first
		forwprop();
		backprop();
		
		FloatMat mat;
		
		// hack: can be a ParamUnit or an InletUnit, 
		// depend on whether this network is a pure compute net or not
		DataUnit w; 
		
		int totalSize = 0;
		float totalGradAbsSum = 0, totalDataAbsSum = 0;
		int i = 0;
		for (ParamUnit param : params)
		{
            // if doesn't have any param, 'params' will only have 1 null, iterate once and exit this loop
			w = hasParams ? param : inlet;
			
			mat = new FloatMat(w.data());
			mat.copyFrom(w.data());
			totalDataAbsSum += mat.abs().sum(); // mat is mutated

			mat.copyFrom(w.gradient());
			totalSize += mat.size() - (hasBias ? mat.row : 0);
			totalGradAbsSum += mat.clone().abs().sum();
			propGrad[i ++] = mat;
		}
		
		// Heuristic to pick a good EPS
		final float EPS = Math.max( totalDataAbsSum / totalSize / perturbRatio, 1e-3f);
		
		// Do finite-diff forward prop for every entry in every parameter
		i = 0;
		for (ParamUnit param : params)
		{
			w = hasParams ? param : inlet;
			mat = new FloatMat(w.data());
					
			for (int idx = 0 ; idx < mat.size(); idx ++)
			{
				// Skip checking the last row if include bias units
				if (hasBias && mat.toCoord(idx).i == mat.row - 1)
					continue;

				// +EPS and -EPS perturb
				float negResult = 0, posResult = 0;
				for (int perturb : new int[] {-1, 1})
				{
    				// Re-init everything as the exact gradient initialization
					this.reset(); inlet.nextBatch();
    				
            		// Perturb -EPS
            		w.data().incrSingle(idx, perturb * EPS);
            		forwprop();
            		float result = terminal.lossTotal();

            		if (perturb < 0) negResult = result; else posResult = result;
				}
				// Compute symmetric numerical gradient and store to 'mat'
				mat.setSingle(idx, (posResult - negResult) / (2 * EPS));
			}
			// Store
			goldGrad[i ++] = mat;
		}
		
		if (verbose)
		{
    		PP.setSep("\n\n");
    		PP.pTitledSectionLine("Back-Prop");
    		PP.p(propGrad);
    		PP.p();
    		PP.pTitledSectionLine("Numerical Gold");
    		PP.p(goldGrad);
    		PP.p();
		}
		PP.pTitledSectionLine("Error Report", "-", 10);
        PP.setSep();
		PP.setPrecision(2); PP.setScientific(true);
		PP.p("\nPerturb EPS = ", EPS);
		// Compare difference
		float absErr = 0;
		for (i = 0 ; i < propGrad.length; i ++)
			absErr += GpuBlas.add(propGrad[i], goldGrad[i], 1, -1).abs().sum();
		float avgAbsErr = absErr / totalSize;
		
		float avgAbsVal = (hasParams ? totalGradAbsSum : totalDataAbsSum) / totalSize;
		float avgPercentErr = avgAbsErr / avgAbsVal * 100;
		
		PP.p("Average absolute error =", avgAbsErr);
		PP.p("Average percent error =", avgPercentErr, "%\n");
		PP.setPrecision(); PP.setScientific(false); // reset printer options
		
		return avgPercentErr;
	}
	
	/**
	 * Default verbose = false
	 */
	public float gradCheck(LearningPlan plan, boolean hasBias, float perturbRatio)
	{
		return this.gradCheck(plan, hasBias, perturbRatio, false);
	}
}