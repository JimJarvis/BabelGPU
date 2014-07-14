package test.deep;

import org.junit.*;
import static test.deep.DeepTestKit.*;
import utils.*;
import deep.*;
import deep.units.*;

public class LinearTest
{
	@BeforeClass
	public static void setUp() { systemInit(); }
	
	@Test
	@Ignore
	public void linearLayersSquareErrorTest()
	{
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 2), 
						new int[] {3, 10, 6, 5, outDim}, 
						SquareErrorUnit.class, 
						Initializer.uniformRandIniter(1));
		linearLayers.name = "Linear + SquareError";
//		linearLayers.runDebug(plan, hasBias);
		check(linearLayers, 3e-4, 1e2f, false);
	}
	
	@Test
	@Ignore
	public void linearLayersCrossEntropyTest()
	{
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 3, InletMode.GoldSumTo1),
						new int[] {5, 8, 6, outDim},
						CrossEntropyUnit.class, 
						Initializer.uniformRandIniter(1));
		linearLayers.name = "Linear + CrossEntropy";
//		linearLayers.runDebug(plan, hasBias);
		check(linearLayers, 1e-2, 1e3f, false);
	}
	
	@Test
//	@Ignore
	public void linearLayersSparseCrossEntropyTest()
	{
		inDim = 3;
		outDim = 3;
		batchSize = 2;
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 0, InletMode.GoldLabel),
						new int[] {4, outDim},
						SparseCrossEntropyUnit.class, 
						Initializer.uniformRandIniter(1));
		linearLayers.name = "Linear + SparseCrossEntropy";
//		linearLayers.runDebug(plan, hasBias);
		check(linearLayers, 1, 1e2f, true);
	}
	
	@Test
	@Ignore
	public void linearLayersSumTest()
	{
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 2), 
						new int[] {3, 10, 6, 5, outDim}, 
						SumUnit.class, 
						Initializer.uniformRandIniter(0,1));
		linearLayers.name = "Linear + Sum";
//		linearLayers.runDebug(plan, hasBiase);
		check(linearLayers, 5e-4, 1e2f, false);
	}
}