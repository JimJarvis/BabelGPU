package test.deep;

import static org.junit.Assert.*;
import gpu.*;

import org.junit.*;

import utils.CpuUtil;
import deep.*;
import deep.units.*;

public class SigmoidNetTest
{
	static InletUnit inlet;
	static LearningPlan plan;
	static int dim, batchSize;
	
	@BeforeClass
	public static void setUp() throws Exception
	{
		GpuBlas.init();

		final float[][] dummyInput = new float[][] {
				{1.5f, 3, -.5f},
				{-0.5f, -1, 2},
				{-2, -4, 2.5f},
				{3, 6, 0}
		};

		final float[][] dummyGold = new float[][] {
				{0, 0, 1},
				{1, 0, 0},
				{0, 1, 0},
				{0, 0, 0}
		};
		
		dim = dummyInput.length;
		batchSize = dummyInput[0].length;

		// totalTrainSize = colDim(input)
		plan = new LearningPlan(2, 1.5f, 0, batchSize);
		
		inlet = new InletUnit("Dummy Inlet", dim, batchSize)
		{
			boolean hasNext = true;
			{
				this.goldMat = new FloatMat(this.data);
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
		DeepNet sigmoidNet = DeepFactory.simpleSigmoidNet(inlet, new int[] {5, 10, 3, 7, 6, dim});
//		sigmoidNet.runDebug(plan);
		check(sigmoidNet, 0.5);
	}
	
	@Test
	public void linearLayersTest()
	{
		DeepNet linearLayers = DeepFactory.debugLinearLayers(inlet, new int[] {3, 5, 2, 1, 4, dim});
//		linearLayers.runDebug(plan);
		check(linearLayers, 5e-3);
	}
	
	@Test
	public void sigmoidLayersTest()
	{
		DeepNet sigmoidLayers = DeepFactory.debugSigmoidLayers(inlet, 1);
//		sigmoidLayers.runDebug(plan);
		check(sigmoidLayers, 2);
	}
}
