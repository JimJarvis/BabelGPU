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
	public void fourierProjectionNetTest()
	{
		DeepNet fourierNet =
				DeepFactory.fourierProjectionNet(
						uniRandInlet(3, 0, InletMode.GoldLabel), 
//						uniformInlet(1, 0), 
						new int[] {5, 4, 10, outDim}, 
						new Initializer[] {Initializer.gaussianProjKernelIniter(3), Initializer.laplacianProjKernelIniter(2), Initializer.cauchyProjKernelIniter(1)},
//						CpuUtil.repeatedArray(Initializer.fillIniter(.1f), 1),
						Initializer.uniformRandIniter(1));
//		fourierNet.runDebug(plan, hasBias);
		check(fourierNet, 1e-2f, 1e2f, false);
	}
	
	@Test
	@Ignore
	public void debugFourierTest()
	{
		InletUnit inlet = uniRandInlet(3, 0, InletMode.GoldLabel);
		int[] projDims = new int[] {5, 10, 6};
		int[] linearDims = new int[] {8, 6, outDim};

		int i;
		ArrayList<ComputeUnit> units = new ArrayList<>();
		for (i = 0; i < projDims.length; i++)
		{
			units.add(new FourierProjectUnit("", projDims[i], Initializer.gaussianProjKernelIniter(2)));
			// scalor = sqrt(2/D) where D is #new features
			units.add(new CosineUnit("", (float) Math.sqrt(2.0 / projDims[i])));
    		units.add(new LinearUnit("", linearDims[i] / 2, Initializer.uniformRandIniter(1)));
			units.add(new FourierProjectUnit("", projDims[i], Initializer.gaussianProjKernelIniter(2)));
			units.add(new SigmoidUnit(""));
    		units.add(new LinearUnit("", linearDims[i], Initializer.uniformRandIniter(1)));
		}
		units.add(new SparseCrossEntropyUnit("", inlet));
		DeepNet net = new DeepNet("DebugFourierNet", inlet, units).genDefaultUnitName();
		
		check(net, 1e-1f, 1e2f, false);
	}
}