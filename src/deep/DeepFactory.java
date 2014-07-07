package deep;

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

		return new DeepNet(inlet, units).genDefaultUnitName();
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
	
	public static DeepNet debugLinearLayers(InletUnit inlet, int[] layerDims)
	{
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerDims.length; i++)
			units.add( new LinearUnit("", layerDims[i], Initializer.uniformRandIniter(1)) );
		units.add(new SquareErrorUnit("", inlet));

		return new DeepNet(inlet, units).genDefaultUnitName();
	}
	
	/**
	 * Stack a couple of sigmoid layers (each has same dim as Inlet)
	 */
	public static DeepNet debugSigmoidLayers(InletUnit inlet, int layerN)
	{
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerN; i++)
			units.add( new SigmoidUnit("") );
		units.add(new SquareErrorUnit("", inlet));

		return new DeepNet(inlet, units).genDefaultUnitName();
	}
}
