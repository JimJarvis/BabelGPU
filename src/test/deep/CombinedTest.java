package test.deep;

import java.util.ArrayList;

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
	@Ignore
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
	public void fourierProjectionNetTest()
	{
		inDim = 7;
		outDim = 7;
		batchSize = 5;
		ArrayList<Initializer> initers = new ArrayList<>();
		initers.add(Initializer.fillIniter(1));
		initers.add(Initializer.fillIniter(2));
		ArrayList<Double> ratios = new ArrayList<>();
		ratios.add(2.);
		ratios.add(1.);
		
		DeepNet fourierNet =
				DeepFactory.fourierProjectionNet(
//						uniRandInlet(3, 0, InletMode.GoldLabel), 
						uniformInlet(1, 0), 
						new int[] {5, outDim}, 
//						new Initializer[] {Initializer.gaussianIniter(2)},
						CpuUtil.repeatedArray(Initializer.fillIniter(.1f), 1),
						Initializer.fillIniter(2));
		fourierNet.runDebug(plan, hasBias);
		check(fourierNet, 1e-1f, 1e2f, true);
	}
}