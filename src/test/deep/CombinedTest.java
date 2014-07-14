package test.deep;

import org.junit.*;
import static test.deep.DeepTestKit.*;
import utils.*;
import deep.*;
import deep.units.*;

public class CombinedTest
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
}