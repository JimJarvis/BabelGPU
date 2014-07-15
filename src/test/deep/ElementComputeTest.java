package test.deep;

import org.junit.*;
import static test.deep.DeepTestKit.*;
import utils.*;
import deep.*;
import deep.units.*;

public class ElementComputeTest
{
	@BeforeClass
	public static void setUp() { systemInit(); }
	
	@Test
//	@Ignore
	public void sigmoidLayersTest()
	{
		DeepNet net = 
				DeepFactory.debugElementComputeLayers(
						SigmoidUnit.class, 
						uniRandInlet(2, 2), 
						2, scalor, 
						SquareErrorUnit.class);
		net.name = "Sigmoid + SquareError";
//		sigmoidLayers.runDebug(plan, hasBias);
		check(net, 1e-1f, 1e2f, false);
	}

	@Test
//	@Ignore
	public void sigmoidLayersCrossEntropyTest()
	{
		DeepNet net = 
				DeepFactory.debugElementComputeLayers(
						SigmoidUnit.class, 
						uniRandInlet(2, 1, InletMode.GoldSumTo1), 
						1, scalor, 
						CrossEntropyUnit.class);
		net.name = "Sigmoid + CrossEntropy";
//		sigmoidLayers.runDebug(plan, hasBias);
		check(net, 1e-1f, 1e2f, false);
	}
	
	@Test
//	@Ignore
	public void cosineLayersTest()
	{
		DeepNet net = 
				DeepFactory.debugElementComputeLayers(
						CosineUnit.class, 
						uniRandInlet(1, 3), 
						3, 1 /*scalor*/, 
						SquareErrorUnit.class);
		net.name = "Cosine + SquareError";
//		cosineLayers.runDebug(plan, hasBias);
		check(net, 1e-1f, 1e2f, false);
	}
	
	@Test
//	@Ignore
	public void cosineLayersCrossEntropyTest()
	{
		DeepNet net = 
				DeepFactory.debugElementComputeLayers(
						CosineUnit.class, 
						uniRandInlet(2, 1, InletMode.GoldSumTo1), 
						2, scalor, 
						CrossEntropyUnit.class);
		net.name = "Cosine + CrossEntropy";
//		sigmoidLayers.runDebug(plan, hasBias);
		check(net, 1e-1f, 1e5f, false);
	}

	@Test
//	@Ignore
	public void cosineLayersSparseCrossEntropyTest()
	{
		DeepNet net = 
				DeepFactory.debugElementComputeLayers(
						CosineUnit.class, 
						uniRandInlet(3, 1, InletMode.GoldLabel), 
						3, scalor, 
						SparseCrossEntropyUnit.class);
		net.name = "Cosine + SparseCrossEntropy";
//		sigmoidLayers.runDebug(plan, hasBias);
		check(net, 1e-1f, 1e5f, false);
	}
}