package deep;

import gpu.FloatMat;

import java.util.ArrayList;

import utils.CpuUtil;
import deep.units.*;

/**
 * Constructs various DeepNets
 */
public class DeepFactory
{
	/**
	 * @param number of neurons in each layer, 
	 * from the first hidden layer to output layer (included)
	 * @param initers for each layer
	 */
	public static DeepNet simpleSigmoidNet(InletUnit inlet, int[] layerDims, Initializer[] initers)
	{
		if (initers.length < layerDims.length)
			throw new DeepException("Not enough initializers for each layer");
		
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerDims.length; i++)
		{
			units.add( new LinearUnit("", layerDims[i], initers[i]) );
			units.add(new SigmoidUnit(""));
		}
		units.add(new SquareErrorUnit("", inlet));

		return 
			new DeepNet("SimpleSigmoidNet", inlet, units).genDefaultUnitName();
	}
	
	/**
	 * @param number of neurons in each layer
	 * from the first hidden layer to output layer (included)
	 * We use the sqrt(6 / (L_in + L_out)) default init scheme
	 */
	public static DeepNet simpleSigmoidNet(InletUnit inlet, int[] layerDims)
	{
		int layerN = layerDims.length;
		Initializer[] initers = new Initializer[layerN];
		for (int i = 0; i < layerN; i++)
		{
			float L_in = i == 0 ? inlet.dim() : layerDims[i - 1];
			float L_out = layerDims[i];
			float symmInitRange = (float) Math.sqrt(6 / (L_in + L_out));
			initers[i] = Initializer.uniformRandIniter(symmInitRange);
		}
		return simpleSigmoidNet(inlet, layerDims, initers);
	}
	
	/**
	 * Use single initer for all layers
	 */
	public static DeepNet simpleSigmoidNet(InletUnit inlet, int[] layerDims, Initializer initer)
	{
		return simpleSigmoidNet(inlet, layerDims, 
				CpuUtil.repeatedArray(initer, layerDims.length));
	}
	
	public static DeepNet debugLinearLayers(
			InletUnit inlet, int[] layerDims, Class<? extends TerminalUnit> terminalClass, Initializer initer)
	{
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerDims.length; i++)
			units.add( new LinearUnit("", layerDims[i], initer) );
		units.add(defaultTerminalCtor(inlet, terminalClass));

		return 
			new DeepNet("DebugLinearLayers", inlet, units).genDefaultUnitName();
	}
	
	/**
	 * Stack a couple of pure computing layers together
	 * Output vector (goldMat) must have the same dim as input vector
	 */
	public static DeepNet debugElementComputeLayers(
			Class<? extends ElementComputeUnit> pureClass, 
			InletUnit inlet, int layerN, 
			Class<? extends TerminalUnit> terminalClass)
	{
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerN; i++)
			units.add( defaultElementComputeCtor(pureClass) );
		units.add(defaultTerminalCtor(inlet, terminalClass));

		return 
			new DeepNet(
				"Debug" + pureClass.getSimpleName().replaceFirst("Unit", "Layers"), inlet, units)
					.genDefaultUnitName();
	}
	
	// Helper to construct terminal units
	private static TerminalUnit defaultTerminalCtor(InletUnit inlet, Class<? extends TerminalUnit> terminalClass)
	{
		try {
			return terminalClass.getConstructor(String.class, InletUnit.class).newInstance("", inlet);
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new DeepException("Terminal class auto-construction fails");
		}
	}
	
	// Helper to construct pure compute layers
	private static ElementComputeUnit defaultElementComputeCtor(
			Class<? extends ElementComputeUnit> pureClass)
	{
		try {
			return pureClass.getConstructor(String.class).newInstance("");
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new DeepException("ElementCompute class auto-construction fails");
		}
	}
}
