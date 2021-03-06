package test.deep;

import static org.junit.Assert.*;
import java.util.Arrays;
import java.util.Random;
import gpu.*;
import utils.*;
import deep.*;
import deep.units.*;

/**
 * Helps configure other unit tests
 * All settings will be global here through the test/deep package
 */
public class DeepTestKit
{
	// ******************** CONFIG ********************/
	// Input feature vector dimension
	public static int inDim = 9;
	// Number of training samples
	public static int batch = 5;
	// Output vector dimension
	public static int outDim = 9;
	// Test option: add bias?
	public static boolean hasBias = true;
	// Scale ElementComputeUnit output
	public static float scalor = 3f;
	// Regularization
	public static float reg = 1.5f;
	public static LearningPlan plan = 
			new LearningPlan("DeepTest", "", 2, reg, batch, 1);
	
	public static GpuRand grand;
	public static Random rand;
	
	public static void systemInit()
	{
		GpuBlas.init();
		grand = new GpuRand(GpuRand.SEED);
		rand = new Random(GpuRand.SEED);
	}
	
	public static int changeDim(int dim)
	{
		return hasBias ? dim + 1 : dim;
	}

	static float[][] dummyInput, dummyGold; static int[] dummyLabels;
	/**
	 * CrossEntropyUnit requires that the sum of each column of goldMat must be 1
	 */
	public static InletUnit uniRandInlet(float inputLow, float inputHigh, float goldLow, float goldHigh, final boolean goldSumTo1)
	{
		dummyInput = 
				CpuUtil.randFloatMat(changeDim(inDim), batch, inputLow, inputHigh);

		dummyGold = 
				CpuUtil.randFloatMat(changeDim(outDim), batch, goldLow, goldHigh);

		dummyLabels = CpuUtil.randInts(batch, outDim);
		
		return new InletUnit("Dummy Inlet", changeDim(inDim) , batch, true)
		{
			{
				this.goldMat = new FloatMat(changeDim(outDim), batch);
				this.goldMat.setHostArray(dummyGold);
				this.goldMat.toDevice(true);
				
				if (hasBias) this.goldMat.fillLastRow0();

				if (goldSumTo1) // normalize col sum to 1 for CrossEntropyUnit
					for (int c = 0; c < goldMat.col; c++)
					{
						FloatMat colMat = goldMat.createColOffset(c);
						GpuBlas.scale(colMat, 1f / colMat.sum());
					}
                this.goldLabels = Thrust.copy_host_to_device(dummyLabels);
			}
			
			@Override
			public void nextGold() { }
			@Override
			protected int nextBatch_()
			{
				this.data.setHostArray(dummyInput);
				this.data.toDevice(true);
				if (hasBias) this.data.fillLastRow1();
				return DeepTestKit.batch;
			}
			@Override
			public void prepareNextEpoch() { }
		};
	}
	
	/**
	 * inputLow, High = +/- inputSymm
	 * goldLow, High = +/- goldSymm
	 * CrossEntropyUnit requires that the sum of each column of goldMat must be 1
	 */
	public static InletUnit uniRandInlet(float inputSymm, float goldSymm, boolean goldSumTo1)
	{
		return uniRandInlet(-inputSymm, inputSymm, -goldSymm, goldSymm, goldSumTo1);
	}
	// default sumTo1 = false
	public static InletUnit uniRandInlet(float inputLow, float inputHigh, float goldLow, float goldHigh)
	{
		return uniRandInlet(inputLow, inputHigh, goldLow, goldHigh, false);
	}
	// default sumTo1 = false
	public static InletUnit uniRandInlet(float inputSymm, float goldSymm)
	{
		return uniRandInlet(-inputSymm, inputSymm, -goldSymm, goldSymm, false);
	}
	
	// Produce an artificial Inlet with uniform input and gold
	public static InletUnit uniformInlet(final float inputVal, final float goldVal)
	{
		return new InletUnit("Uniform Inlet", changeDim(inDim), batch, true)
		{
			{
				this.goldMat = new FloatMat(changeDim(outDim), batch);
				this.goldMat.fill(goldVal);
				if (hasBias) this.goldMat.fillLastRow0();
				this.goldLabels = 
						Thrust.copy_host_to_device(CpuUtil.randInts(batch, outDim));
			}

			@Override
			public void nextGold() { }
			@Override
			protected int nextBatch_()
			{
				this.data.fill(inputVal);
				if (hasBias) this.data.fillLastRow1();
				return DeepTestKit.batch;
			}
			@Override
			public void prepareNextEpoch() { }
		};
	}
	
	/**
	 * @param TOL within tolerance percentage (already multiplied by 100)
	 * @param perturbRatio @see DeepNet#gradCheck()
	 * When debugging PureComputeLayers, make sure inDim == outDim
	 */
	public static void check(DeepNet net, double TOL, float perturbRatio, boolean verbose)
	{
		if (net.getParamList().size() == 0// this is a PureCompute debug network 
			// the terminal class requires inDim == outDim
			&& Arrays.asList(SquareErrorTUnit.class, CrossEntropyTUnit.class).contains(net.terminal.getClass())
			&& inDim != outDim)
		{
			fail("PureComputeLayer debug test must have inDim == outDim");
		}
		
		float avgPercentErr = net.gradCheck(plan, hasBias, perturbRatio, verbose);
		assertEquals(net.name + " grad check", avgPercentErr, 0, TOL);
	}
	public static void check(DeepNet net, double TOL, float perturbRatio) {	check(net, TOL, perturbRatio, false);	}
	public static void check(DeepNet net, double TOL, boolean verbose) {	check(net, TOL, 1e3f, verbose);	}
	public static void check(DeepNet net, double TOL) {	check(net, TOL, 1e3f, false);	}
}