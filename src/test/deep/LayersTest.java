package test.deep;

import static org.junit.Assert.*;
import org.junit.*;
import java.util.Arrays;
import gpu.*;
import utils.*;
import deep.*;
import deep.units.*;

public class LayersTest
{
	// ******************** CONFIG ********************/
	// Input feature vector dimension
	static int inDim = 9;
	// Number of training samples
	static int batchSize = 5;
	// Output vector dimension
	static int outDim = 9;
	// Test option: add bias?
	static boolean hasBias = true;
	// Scale ElementComputeUnit output
	static float scalor = 3f;
	
	// Regularization
	static float reg = 1.5f;
	static LearningPlan plan = new LearningPlan(2, reg, 0, batchSize);
	
	@BeforeClass
	public static void setUp() throws Exception
	{
		GpuBlas.init();
		GpuUtil.enableExceptions();
	}
	
	private static int changeDim(int dim)
	{
		return hasBias ? dim + 1 : dim;
	}

	public InletUnit uniRandInlet(float inputLow, float inputHigh, float goldLow, float goldHigh)
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
	public InletUnit uniRandInlet(float inputSymm, float goldSymm)
	{
		return uniRandInlet(-inputSymm, inputSymm, -goldSymm, goldSymm);
	}
	
	// Produce an artificial Inlet with uniform input and gold
	public InletUnit uniformInlet(final float inputVal, final float goldVal)
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
	private void check(DeepNet net, double TOL, float perturbRatio, boolean verbose)
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
	private void check(DeepNet net, double TOL, float perturbRatio) {	check(net, TOL, perturbRatio, false);	}
	private void check(DeepNet net, double TOL, boolean verbose) {	check(net, TOL, 1e3f, verbose);	}
	private void check(DeepNet net, double TOL) {	check(net, TOL, 1e3f, false);	}
	
	@Test
//	@Ignore
	public void simpleSigmoidNetTest()
	{
		DeepNet sigmoidNet = 
				DeepFactory.simpleSigmoidNet(
						uniRandInlet(-2, 2, 0, 1), 
						new int[] {5, 10, 3, 6, outDim});
//		sigmoidNet.runDebug(plan);
		check(sigmoidNet, 1e-1f, 1e2f, false);
	}
	
	@Test
//	@Ignore
	public void linearLayersSquareErrorTest()
	{
		PP.p("LINEAR: SquareError");
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 2), 
						new int[] {3, 10, 6, 5, outDim}, 
						SquareErrorUnit.class, 
						Initializer.uniformRandIniter(1));
//		linearLayers.runDebug(plan, hasBias);
		check(linearLayers, 3e-4, 1e2f, false);
	}
	
	@Test
//	@Ignore
	public void linearLayersSumTest()
	{
		PP.p("LINEAR: Sum");
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 2), 
						new int[] {3, 10, 6, 5, outDim}, 
						SumUnit.class, 
						Initializer.uniformRandIniter(0,1));
//		linearLayers.runDebug(plan, hasBiase);
		check(linearLayers, 5e-4, 1e2f, false);
	}

	@Test
//	@Ignore
	public void sigmoidLayersTest()
	{
		DeepNet sigmoidLayers = 
				DeepFactory.debugElementComputeLayers(
						SigmoidUnit.class, 
						uniRandInlet(2, 2), 
						2, scalor, 
						SquareErrorUnit.class);
//		sigmoidLayers.runDebug(plan, hasBias);
		check(sigmoidLayers, 1e-1f, 1e1f, false);
	}
	
	@Test
//	@Ignore
	public void cosineLayersTest()
	{
		DeepNet cosineLayers = 
				DeepFactory.debugElementComputeLayers(
						CosineUnit.class, 
						uniRandInlet(1, 1), 
						3, scalor, 
						SquareErrorUnit.class);
//		cosineLayers = 
//				DeepFactory.debugElementComputeLayers(CosineUnit.class, uniformInlet(0.6f, 0.5f), 4, scalor, SumUnit.class);
//		cosineLayers.runDebug(plan, hasBias);
		check(cosineLayers, 1e-1f, 1e5f, false);
	}
}