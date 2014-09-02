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
//	@Ignore
	public void linearLayersSquareErrorTest()
	{
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 2), 
						new int[] {3, 10, 6, 5, outDim}, 
						SquareErrorTUnit.class, 
						Initializer.uniformRandIniter(1));
		linearLayers.name = "Linear + SquareError";
//		linearLayers.runDebug(plan, hasBias);
		check(linearLayers, 5e-4, 1e2f, false);
	}
	
	@Test
//	@Ignore
	public void linearLayersCrossEntropyTest()
	{
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 3, true),
						new int[] {5, 6, outDim},
						CrossEntropyTUnit.class, 
						Initializer.uniformRandIniter(1));
		linearLayers.name = "Linear + CrossEntropy";
//		linearLayers.runDebug(plan, hasBias);
		check(linearLayers, 5e-2, 1e2f, false);
	}
	
	@Test
//	@Ignore
	public void linearLayersSparseCrossEntropyTest()
	{
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(1, 0, true),
						new int[] {8, 3, outDim},
						SparseCrossEntropyTUnit.class, 
						Initializer.uniformRandIniter(1));
		linearLayers.name = "Linear + SparseCrossEntropy";
//		linearLayers.runDebug(plan, hasBias);
		check(linearLayers, 5e-2, 1e2f, false);
	}
	
	@Test
//	@Ignore
	public void linearLayersSumTest()
	{
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 2), 
						new int[] {3, 10, 6, 5, outDim}, 
						SumTUnit.class, 
						Initializer.uniformRandIniter(0,1));
		linearLayers.name = "Linear + Sum";
//		linearLayers.runDebug(plan, hasBiase);
		check(linearLayers, 5e-4, 1e2f, false);
	}
}