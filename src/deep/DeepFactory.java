package deep;

import java.util.ArrayList;

import utils.*;
import deep.units.*;

/**
 * Constructs various DeepNets
 */
public class DeepFactory
{
	/**
	 * @param number of neurons in each layer, 
	 * from the first hidden layer to output layer (included)
	 * @param initers for each layer. If only 1 initer, we repeat it for all layers
	 */
	public static DeepNet simpleForwardNet(
			InletUnit inlet, int[] layerDims, 
			Class<? extends ElementComputeUnit>[] activationLayers, 
			Initializer... initers)
	{
		int layerN = layerDims.length;
		
		if (initers.length == 1) // repeat for all layers
			initers = MiscUtil.repeatedArray(initers[0], layerN);
		else if (initers.length < layerN)
			throw new DeepException("Not enough initializers for each layer");
		
		if (activationLayers.length != layerN - 1)
			throw new DeepException(
					"Number of activation layers should be 1 less than number of layerDims, "
					+ "because the last layer will always be SparseCrossEntropy");

		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerN; i++)
		{
			units.add(new LinearUnit("", inlet, layerDims[i], initers[i]) );
			if (i != layerN - 1) // if not the last
                units.add(defaultElementComputeCtor(activationLayers[i], inlet, 1));
		}
		units.add(new SparseCrossEntropyTUnit("", inlet));

		return 
			new DeepNet("SimpleForwardNet", inlet, units).genDefaultUnitName();
	}

	/**
	 * @param number of neurons in each layer
	 * from the first hidden layer to output layer (included)
	 * We use the sqrt(6 / (L_in + L_out)) default init scheme
	 */
	public static DeepNet simpleForwardNet(InletUnit inlet, 
			Class<? extends ElementComputeUnit>[] activationLayers, 
			int ... layerDims)
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
		return simpleForwardNet(inlet, layerDims, activationLayers, initers);
	}

	/**
	 * Repeated activation layers with sqrt(3/(L1 + L2)) initer
	 */
	public static DeepNet simpleForwardNet(InletUnit inlet, 
			Class<? extends ElementComputeUnit> activationLayer, 
			int ... layerDims)
	{
		return simpleForwardNet(inlet,
				MiscUtil.repeatedArray(activationLayer, layerDims.length - 1),
				layerDims);
	}

	/**
	 * Repeated activation layers with a single repeated initer
	 */
	public static DeepNet simpleForwardNet(InletUnit inlet, 
			Class<? extends ElementComputeUnit> activationLayer, 
			Initializer initer, int ... layerDims)
	{
		return simpleForwardNet(inlet, layerDims, 
				MiscUtil.repeatedArray(activationLayer, layerDims.length - 1),
				MiscUtil.repeatedArray(initer, layerDims.length));
	}
	

	public static DeepNet simpleSigmoidNet(InletUnit inlet, int ... layerDims)
	{
		return simpleForwardNet(inlet, 
				MiscUtil.repeatedArray(SigmoidUnit.class, layerDims.length - 1), layerDims);
	}
	
	/**
	 * Add one more layer to simple forward net
	 * @param initZero true to init all to zero. False to init to random numbers. 
	 */
	public static DeepNet growSimpleForwardNet(
			DeepNet oldNet, int newLayerDim, 
			Class<? extends ElementComputeUnit> activationLayer,
			boolean initZero)
	{
		ArrayList<ComputeUnit> units = oldNet.getUnitList();
		TerminalUnit terminal = (TerminalUnit) MiscUtil.get(units, -1);
		InletUnit inlet = oldNet.inlet;
		// add sigmoid linear layer
		MiscUtil.set(units, -1, defaultElementComputeCtor(activationLayer, inlet, 1));
		units.add(new LinearUnit("", inlet, newLayerDim, 
				initZero ? Initializer.fillIniter(0) :
					Initializer.uniformRandIniter((float) Math.sqrt(3.0 / newLayerDim))));
		units.add(terminal);
		
		DeepNet newNet = 
			new DeepNet("SimpleForwardNet", inlet, units).genDefaultUnitName();
		
		return newNet;
	}
	
	public static DeepNet growSimpleSigmoidNet(DeepNet oldNet, int newLayerDim, boolean initZero)
	{
		return growSimpleForwardNet(oldNet, newLayerDim, SigmoidUnit.class, initZero);
	}
	
	
	/**
	 * Add one more layer to fourier stack net
	 * @param initZero true to init all to zero. False to init to random numbers. 
	 */
	public static DeepNet growFourierStackNet(
			DeepNet oldNet, int newFourierDim, int newLinearDim, 
			Initializer projIniter,
//			Class<? extends ElementComputeUnit> activationLayer,
			boolean initZero)
	{
		ArrayList<ComputeUnit> units = oldNet.getUnitList();
		TerminalUnit terminal = (TerminalUnit) MiscUtil.get(units, -1);
		InletUnit inlet = oldNet.inlet;
		// add sigmoid linear layer
		MiscUtil.set(units, -1, new FourierProjectUnit("", inlet, newFourierDim, projIniter));
		units.add(new CosineUnit("", inlet, (float) Math.sqrt(2.0 / newFourierDim)));
		units.add(new LinearUnit("", inlet, newLinearDim, 
				initZero ? Initializer.fillIniter(0) :
					Initializer.uniformRandIniter((float) Math.sqrt(3.0 / newLinearDim))));
		units.add(terminal);
		
		DeepNet newNet = 
			new DeepNet("FourierStackNet", inlet, units).genDefaultUnitName();
		
		return newNet;
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
			units.add(new FourierProjectUnit("", inlet, layerDims[i], projIniters[i]));
			// scalor = sqrt(2/D) where D is #new features
			units.add(new CosineUnit("", inlet, (float) Math.sqrt(2.0 / layerDims[i])));
		}
		units.add(new LinearUnit("", inlet, layerDims[i], lastLinearIniter));
		units.add(new SparseCrossEntropyTUnit("", inlet));
		return 
			new DeepNet("FourierProjectionNet", inlet, units).genDefaultUnitName();
	}
	
	/**
	 * Compute-only Fourier projection net
	 * The layers will all be Rahimi projectors
	 * The last one will be a dummy ForwardOnly terminal
	 * @param layerDims for projectors
	 * @param activationLayers ElementComputeUnits
	 * @param projIniters must match layerDims in length
	 */
	public static DeepNet fourierProjectionNet(
			InletUnit inlet, int[] layerDims, 
			Class<? extends ElementComputeUnit>[] activationLayers, 
			Initializer... projIniters)
	{
		if (projIniters.length != layerDims.length)
			throw new DeepException("Exactly 1 projection initer for each layer: len(projIniter)==len(layerDims)");
		
		ArrayList<ComputeUnit> units = new ArrayList<>();
		int i;
		ElementComputeUnit eleUnit;
		for (i = 0; i < layerDims.length; i++)
		{
			units.add(new FourierProjectUnit("", inlet, layerDims[i], projIniters[i]));
//			eleUnit = new CosineUnit("", inlet, (float) Math.sqrt(2.0 / layerDims[i]));
			// scalor = sqrt(2/D) where D is #new features
			eleUnit = defaultElementComputeCtor(
					activationLayers[i], inlet, 
					(float) Math.sqrt(2.0 / layerDims[i]));
			eleUnit.setMergeIO(true);
			units.add(eleUnit);
		}
		units.add(new ForwardOnlyTUnit("", inlet));
		return 
			new DeepNet("FourierProjectionNet(foward-only)", inlet, units).genDefaultUnitName();
	}
	
	/**
	 * Defaults to CosineUnit activations
	 * @see #fourierProjectionNet(InletUnit, int[], CosineUnit[], Initializer...)
	 */
	public static DeepNet fourierProjectionNet(
			InletUnit inlet, int[] layerDims, Initializer... projIniters)
	{
		return fourierProjectionNet(inlet, layerDims, 
				MiscUtil.repeatedArray(CosineUnit.class, layerDims.length), 
				projIniters);
	}
	
	public static DeepNet debugLinearLayers(
			InletUnit inlet, int[] layerDims, Class<? extends TerminalUnit> terminalClass, Initializer initer)
	{
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerDims.length; i++)
			units.add( new LinearUnit("", inlet, layerDims[i], initer) );
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
			units.add( defaultElementComputeCtor(pureClass, inlet, scalor) );
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
			return terminalClass
					.getConstructor(String.class, InletUnit.class)
					.newInstance("", inlet);
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new DeepException("Terminal class auto-construction fails");
		}
	}
	
	// Helper to construct pure compute layers
	private static ElementComputeUnit defaultElementComputeCtor(
			Class<? extends ElementComputeUnit> pureClass, InletUnit inlet, float scalor)
	{
		try {
			return pureClass
					.getConstructor(String.class, InletUnit.class, float.class)
					.newInstance("", inlet, scalor);
		}
		catch (Exception e)
		{
			e.printStackTrace();
			throw new DeepException("ElementCompute class auto-construction fails");
		}
	}
}
