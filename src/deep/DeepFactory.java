package deep;

import java.util.ArrayList;

import deep.units.*;

/**
 * Constructs various DeepNets
 */
public class DeepFactory
{
	/**
	 * @param number of neurons in each layer, 
	 * from the first hidden layer to output layer (included)
	 * @param initializers for each layer
	 */
	public static DeepNet simpleSigmoidNet(InletUnit inlet, int[] layerDims, Initializer[] initializers)
	{
		if (initializers.length < layerDims.length)
			throw new DeepException("Not enough initializers for each layer");
		
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerDims.length; i++)
		{
			units.add( new LinearUnit("", layerDims[i], initializers[i]) );
			units.add(new SigmoidUnit(""));
		}
		units.add(new SquareErrorUnit("", inlet));

		return new DeepNet("SimpleSigmoidNet", inlet, units).genDefaultUnitName();
	}
	
	/**
	 * @param number of neurons in each layer
	 * from the first hidden layer to output layer (included)
	 * We use the sqrt(6 / (L_in + L_out)) default init scheme
	 */
	public static DeepNet simpleSigmoidNet(InletUnit inlet, int[] layerDims)
	{
		int layerN = layerDims.length;
		Initializer[] initializers = new Initializer[layerN];
		for (int i = 0; i < layerN; i++)
		{
			float L_in = i == 0 ? inlet.dim() : layerDims[i - 1];
			float L_out = layerDims[i];
			float symmInitRange = (float) Math.sqrt(6 / (L_in + L_out));
			initializers[i] = Initializer.uniformRandIniter(symmInitRange);
		}
		return simpleSigmoidNet(inlet, layerDims, initializers);
	}
	
	// Helper to construct debug terminal units
	private static TerminalUnit defaultTerminalCtor(InletUnit inlet, Class<? extends TerminalUnit> terminalClass)
	{
		try
		{
			return terminalClass.getConstructor(String.class, InletUnit.class).newInstance("", inlet);
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new DeepException("Terminal class auto-construction fails");
		}
	}
	
	public static DeepNet debugLinearLayers(
			InletUnit inlet, int[] layerDims, Class<? extends TerminalUnit> terminalClass)
	{
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerDims.length; i++)
			units.add( new LinearUnit("", layerDims[i], Initializer.uniformRandIniter(1)) );
		units.add(defaultTerminalCtor(inlet, terminalClass));

		return new DeepNet("DebugLinearLayers", inlet, units).genDefaultUnitName();
	}
	
	/**
	 * Stack a couple of sigmoid layers (each has same dim as Inlet)
	 */
	public static DeepNet debugSigmoidLayers(
			InletUnit inlet, int layerN, Class<? extends TerminalUnit> terminalClass)
	{
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerN; i++)
			units.add( new SigmoidUnit("") );
		units.add(defaultTerminalCtor(inlet, terminalClass));

		return new DeepNet("DebugSigmoidLayers", inlet, units).genDefaultUnitName();
	}
}
