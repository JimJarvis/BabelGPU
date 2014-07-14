package test.deep;

import static org.junit.Assert.*;
import org.junit.*;
import static test.deep.DeepTestKit.*;
import utils.*;
import deep.*;
import deep.units.*;

public class SimpleSigmoidNetTest
{
	@BeforeClass
	public static void setUp() { systemInit(); }
	
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
	public void linearLayersCrossEntropyTest()
	{
		PP.p("LINEAR: Softmax-CrossEntropy");
		DeepNet linearLayers = 
				DeepFactory.debugLinearLayers(
						uniRandInlet(2, 3, true),
						new int[] {5, 8, 6, outDim},
						CrossEntropyUnit.class, 
						Initializer.uniformRandIniter(1));
//		linearLayers.runDebug(plan, hasBias);
		check(linearLayers, 1e-2, 1e2f, false);
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
}