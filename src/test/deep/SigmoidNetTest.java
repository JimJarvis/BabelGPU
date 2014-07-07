package test.deep;

import static org.junit.Assert.*;
import gpu.*;

import org.junit.*;

import utils.CpuUtil;
import utils.GpuUtil;
import deep.*;
import deep.units.*;

public class SigmoidNetTest
{
	// ******************** CONFIG ********************/
	// Input feature vector dimension
	static int inDim = 40;
	// Number of training samples
	static int batchSize = 30;
	// Output vector dimension
	static int outDim = 40;
	// Test option: use random generated numbers?
	static boolean randInput = true; 
	
	// Regularization
	static float reg = 1.5f;
	static LearningPlan plan;
	
	static InletUnit inlet;
	
	@BeforeClass
	public static void system()
	{
		GpuBlas.init();
		GpuUtil.enableExceptions();
	}

	@BeforeClass
	public static void setUp() throws Exception
	{
		// for random input/gold gen
		float symmRange = 2;
		
		final float[][] dummyInput = randInput ? 
    		CpuUtil.randFloatMat(inDim, batchSize, -symmRange, symmRange) :
    		new float[][] {
    				{.5f, 1, -.5f},
    				{-1.5f, -.7f, .6f},
    				{-2, -3, 1.1f},
    				{.3f, -1.6f, 2.1f}
    		};

		final float[][] dummyGold = randInput ? 
    		CpuUtil.randFloatMat(outDim, batchSize, 0, 1) :
    		new float[][] {
    				{0, 0, 1},
    				{1, 0, 0},
    				{0, 1, 0},
    				{0, 0, 1},
    				{1, 0, 0}
    		};
		
    	if (!randInput)
    	{
    		inDim = dummyInput.length;
    		batchSize = dummyInput[0].length;
    		outDim = dummyGold.length;
    	}
		
		// totalTrainSize = colDim(input)
		plan = new LearningPlan(2, reg, 0, batchSize);

		inlet = new InletUnit("Dummy Inlet", inDim, batchSize)
		{
			boolean hasNext = true;
			{
				this.goldMat = new FloatMat(outDim, batchSize);
			}
			
			@Override
			public void nextGold()
			{
				this.goldMat.setHostArray(dummyGold);
				this.goldMat.toDevice(true);
			}
			
			@Override
			public void nextBatch()
			{
				this.data.setHostArray(dummyInput);
				this.data.toDevice(true);
				hasNext = false;
			}
			
			@Override
			public boolean hasNext()
			{
				return hasNext;
			}

			@Override
			public void reset()
			{
				hasNext = true;
			}
		};
		
	}
	
	/**
	 * @param TOL within tolerance percentage (already multiplied by 100)
	 */
	private void check(DeepNet net, double TOL)
	{
		float avgPercentErr = net.gradCheck(plan);
		assertTrue(CpuUtil.withinTol(avgPercentErr, 0, TOL));
	}

	@Test
	@Ignore
	public void simpleSigmoidNetTest()
	{
		DeepNet sigmoidNet = DeepFactory.simpleSigmoidNet(inlet, new int[] {5, 3, 6, outDim});
//		sigmoidNet.runDebug(plan);
		check(sigmoidNet, 0.5);
	}
	
	@Test
	@Ignore
	public void linearLayersTest()
	{
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(inlet, new int[] {3, 5, 6, 4, outDim}, SquareErrorUnit.class);
//		linearLayers.runDebug(plan);
		check(linearLayers, 5e-3);
	}
	
	@Test
	public void sigmoidLayersTest()
	{
		DeepNet sigmoidLayers = DeepFactory.debugSigmoidLayers(inlet, 1, SquareErrorUnit.class);
		sigmoidLayers.runDebug(plan);
		check(sigmoidLayers, 1);
	}
}
