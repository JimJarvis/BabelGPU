package test.deep;

import org.junit.*;
import static test.deep.DeepTestKit.*;
import utils.*;
import deep.*;
import deep.units.*;

public class MiscTest
{
	@BeforeClass
	public static void setUp() { systemInit(); }
	
	@Test
//	@Ignore
	public void growSimpleNetTest()
	{
		Initializer.resetRand(2266400);
		InletUnit inlet = uniRandInlet(2, 2);
		DeepNet net = DeepFactory.simpleSigmoidNet(inlet, 5, 8);
		net.setup(new LearningPlan("", null, 3, 1, 100, 10));
		inlet.nextBatch();
		net.forwprop();
		net.backprop();
		PP.p(net.getParamList());
		PP.pSectionLine();
		PP.pTitledSectionLine("NEW");
		net = DeepFactory.growSimpleSigmoidNet(net, 3);
		net.setup(new LearningPlan("", null, 3, 1, 100, 10));
		PP.p(net.getParamList());
	}
}	