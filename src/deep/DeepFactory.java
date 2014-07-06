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
	 */
	public static DeepNet simpleSigmoidNet(InletUnit inlet, int[] layerDims)
	{
		int layerN = layerDims.length;
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (int i = 0; i < layerN; i++)
		{
			// We use the sqrt(6 / (L_in + L_out)) scheme)
			float L_in = i == 0 ? inlet.dim() : layerDims[i - 1];
			float L_out = layerDims[i];
			float symmInitRange = (float) Math.sqrt(6 / (L_in + L_out));
			units.add( new LinearUnit("", layerDims[i], 
								Initializer.uniformRandInitializer(symmInitRange)) );
			units.add(new SigmoidUnit(""));
		}
		units.add(new SquareErrorUnit("", inlet));

		units.get(0).input = inlet;

		return new DeepNet(units);
	}
}
