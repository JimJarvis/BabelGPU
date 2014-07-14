package test.deep;

import static org.junit.Assert.*;
import org.junit.*;
import static test.deep.DeepTestKit.*;
import utils.*;
import deep.*;
import deep.units.*;

public class ElementComputeLayersTest
{
	@BeforeClass
	public static void setUp() { systemInit(); }
	
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
	@Ignore
	public void sigmoidLayersCrossEntropyTest()
	{
		DeepNet sigmoidLayers = 
				DeepFactory.debugElementComputeLayers(
						SigmoidUnit.class, 
						uniRandInlet(2, 2), 
						2, scalor, 
						SquareErrorUnit.class);
//		sigmoidLayers.runDebug(plan, hasBias);
		check(sigmoidLayers, 1e-1f, 1e1f, true);
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
//		cosineLayers.runDebug(plan, hasBias);
		check(cosineLayers, 1e-1f, 1e5f, false);
	}
}