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
	public static DeepNet simpleSigmoidNet(InletUnit inlet, int[] layerDims, Initializer... initers)
	{
		if (initers.length < layerDims.length)
			throw new DeepException("Not enough initializers for each layer");
		
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerDims.length; i++)
		{
			units.add( new LinearUnit("", layerDims[i], initers[i]) );
			units.add(new SigmoidUnit(""));
		}
		units.add(new SquareErrorTUnit("", inlet));

		return 
			new DeepNet("SimpleSigmoidNet", inlet, units).genDefaultUnitName();
	}
	
	/**
	 * @param number of neurons in each layer
	 * from the first hidden layer to output layer (included)
	 * We use the sqrt(6 / (L_in + L_out)) default init scheme
	 */
	public static DeepNet simpleSigmoidNet(InletUnit inlet, int ... layerDims)
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
	
	/**
	 * The first (layerDims.length - 1) layers will all be Rahimi projectors
	 * The last one will be SparseCrossEntropy terminal
	 * @param layerDims including the terminal unit
	 * @param projIniters for the first few projection layers
	 * @param lastLinearIniter for the last linear layer before terminal
	 */
	public static DeepNet fourierProjectionNet(
			InletUnit inlet, int[] layerDims, Initializer[] projIniters, Initializer lastLinearIniter)
	{
		if (projIniters.length != layerDims.length - 1)
			throw new DeepException("Exactly 1 projection initer for each layer: len(projIniter)==len(layerDims)-1");
		
		ArrayList<ComputeUnit> units = new ArrayList<>();
		int i;
		for (i = 0; i < layerDims.length - 1; i++)
		{
			units.add(new FourierProjectUnit("", layerDims[i], projIniters[i]));
			// scalor = sqrt(2/D) where D is #new features
			units.add(new CosineUnit("", (float) Math.sqrt(2.0 / layerDims[i])));
		}
		units.add(new LinearUnit("", layerDims[i], lastLinearIniter));
		units.add(new SparseCrossEntropyTUnit("", inlet));
		return 
			new DeepNet("FourierProjectionNet", inlet, units).genDefaultUnitName();
	}
	
	/**
	 * Compute-only Fourier projection net
	 * The layers will all be Rahimi projectors
	 * The last one will be a dummy ForwardOnly terminal
	 * @param layerDims for projectors
	 * @param projIniters must match layerDims in length
	 */
	public static DeepNet fourierProjectionNet(
			InletUnit inlet, int[] layerDims, Initializer... projIniters)
	{
		if (projIniters.length != layerDims.length)
			throw new DeepException("Exactly 1 projection initer for each layer: len(projIniter)==len(layerDims)");
		
		ArrayList<ComputeUnit> units = new ArrayList<>();
		int i;
		ElementComputeUnit eleUnit;
		for (i = 0; i < layerDims.length; i++)
		{
			units.add(new FourierProjectUnit("", layerDims[i], projIniters[i]));
			// scalor = sqrt(2/D) where D is #new features
			eleUnit = new CosineUnit("", (float) Math.sqrt(2.0 / layerDims[i]));
			eleUnit.setMergeIO(true);
			units.add(eleUnit);
		}
		units.add(new ForwardOnlyTUnit(""));
		return 
			new DeepNet("FourierProjectionNet(foward-only)", inlet, units).genDefaultUnitName();
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
	 * @param scalor scales each ElementComputeUnit output by a constant
	 */
	public static DeepNet debugElementComputeLayers(
			Class<? extends ElementComputeUnit> pureClass, 
			InletUnit inlet, int layerN, float scalor,
			Class<? extends TerminalUnit> terminalClass)
	{
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerN; i++)
			units.add( defaultElementComputeCtor(pureClass, scalor) );
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
			Class<? extends ElementComputeUnit> pureClass, float scalor)
	{
		try {
			return pureClass.getConstructor(String.class, float.class).newInstance("", scalor);
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new DeepException("ElementCompute class auto-construction fails");
		}
	}
}
