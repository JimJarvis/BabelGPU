package test.deep;

import static org.junit.Assert.*;
import java.util.Arrays;
import gpu.*;
import utils.*;
import deep.*;
import deep.units.*;

/**
 * Helps configure other unit tests
 * All settings will be global here through the test/deep package
 */
public class TestKit
{
	// ******************** CONFIG ********************/
	// Input feature vector dimension
	public static int inDim = 9;
	// Number of training samples
	public static int batchSize = 5;
	// Output vector dimension
	public static int outDim = 9;
	// Test option: add bias?
	public static boolean hasBias = true;
	// Scale ElementComputeUnit output
	public static float scalor = 3f;
	
	// Regularization
	public static float reg = 1.5f;
	public static LearningPlan plan = new LearningPlan(2, reg, 0, batchSize);
	
	public static void systemInit()
	{
		GpuBlas.init();
		GpuUtil.enableExceptions();
	}
	
	public static int changeDim(int dim)
	{
		return hasBias ? dim + 1 : dim;
	}

	public static InletUnit uniRandInlet(float inputLow, float inputHigh, float goldLow, float goldHigh)
	{
		// inDim and outDim + 1 because of bias units
		final float[][] dummyInput = 
				CpuUtil.randFloatMat(inDim + 1, batchSize, inputLow, inputHigh);

		final float[][] dummyGold = 
				CpuUtil.randFloatMat(outDim + 1, batchSize, goldLow, goldHigh);

		return new InletUnit("Dummy Inlet", changeDim(inDim) , batchSize)
		{
			boolean hasNext = true;
			{
				this.goldMat = new FloatMat(changeDim(outDim), batchSize);
			}
			@Override
			public void nextGold()
			{
				this.goldMat.setHostArray(dummyGold);
				this.goldMat.toDevice(true);
				if (hasBias) this.goldMat.fillRow(0, -1);
			}
			@Override
			public void nextBatch()
			{
				this.data.setHostArray(dummyInput);
				this.data.toDevice(true);
				if (hasBias) this.data.fillRow(1, -1);
				hasNext = false;
			}
			@Override
			public boolean hasNext() { return hasNext; }
			@Override
			public void reset() { hasNext = true; }
		};
	}
	
	/**
	 * inputLow, High = +/- inputSymm
	 * goldLow, High = +/- goldSymm
	 */
	public static InletUnit uniRandInlet(float inputSymm, float goldSymm)
	{
		return uniRandInlet(-inputSymm, inputSymm, -goldSymm, goldSymm);
	}
	
	// Produce an artificial Inlet with uniform input and gold
	public static InletUnit uniformInlet(final float inputVal, final float goldVal)
	{
		return new InletUnit("Uniform Inlet", changeDim(inDim), batchSize)
		{
			boolean hasNext = true;
			{
				this.goldMat = new FloatMat(changeDim(outDim), batchSize);
			}
			@Override
			public void nextGold()
			{
				this.goldMat.fill(goldVal);
				if (hasBias) this.goldMat.fillRow(0, -1);
			}
			@Override
			public void nextBatch()
			{
				this.data.fill(inputVal);
				if (hasBias) this.data.fillRow(1, -1);
				hasNext = false;
			}
			@Override
			public boolean hasNext() { return hasNext; }
			@Override
			public void reset() { hasNext = true; }
		};
	}
	
	/**
	 * @param TOL within tolerance percentage (already multiplied by 100)
	 * @param perturbRatio @see DeepNet#gradCheck()
	 * When debugging PureComputeLayers, make sure inDim == outDim
	 */
	public static void check(DeepNet net, double TOL, float perturbRatio, boolean verbose)
	{
		if (net.getParams().size() == 0// this is a PureCompute debug network 
			// the terminal class requires inDim == outDim
			&& Arrays.asList( new Class[] {SquareErrorUnit.class}).contains(net.terminal.getClass())
			&& inDim != outDim)
		{
			fail("PureComputeLayer debug test must have inDim == outDim");
		}
		
		float avgPercentErr = net.gradCheck(plan, hasBias, perturbRatio, verbose);
		assertTrue(net.name + " grad check", CpuUtil.withinTol(avgPercentErr, 0, TOL));
	}
	public static void check(DeepNet net, double TOL, float perturbRatio) {	check(net, TOL, perturbRatio, false);	}
	public static void check(DeepNet net, double TOL, boolean verbose) {	check(net, TOL, 1e3f, verbose);	}
	public static void check(DeepNet net, double TOL) {	check(net, TOL, 1e3f, false);	}
}