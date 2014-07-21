package test.deep;

import org.junit.*;
import gpu.*;
import java.util.ArrayList;
import static test.deep.DeepTestKit.*;
import static org.junit.Assert.*;
import utils.*;
import deep.*;
import deep.Initializer.ProjKernel;
import deep.units.*;

public class KernelApproxTest
{
	static int origDim = 20;
	static int newDim = 50000;
	static int batchSize = 100;
	static float gamma = .1f; // common
	static double TOL = 2;

	// Contains original features
	static FloatMat origData;
	// Holds newly generated features
	static FloatMat newData;
	// newData = projector * origData
	static FloatMat projector;
	
	@BeforeClass
	public static void setUp()
	{ 
		systemInit();
		// We test case where hasBias is true
		origData = grand.genUniformFloat(origDim + 1, batchSize, 0, 2);
		origData.fillLastRow1();
		newData = new FloatMat(newDim + 1, batchSize);
		projector = new FloatMat(newDim + 1, origDim + 1);
	}
	
	/**
	 * Implement the abstract method to test various kernel approximations
	 */
	abstract class KernelDebugger
	{
		protected float gamma;
		private ProjKernel kernelType;
		public KernelDebugger(ProjKernel kernelType, float gamma)
		{
			this.gamma = gamma;
			this.kernelType = kernelType;
			Initializer projIniter = Initializer.projKernelIniter(kernelType, gamma);
			projIniter.setBias(true);
			projIniter.init(projector);
			GpuBlas.scale(
					GpuBlas.mult(projector, origData, newData).cos(), 
					(float) Math.sqrt(2.0 / newDim));
			newData.fillLastRow0();
		}
		
		// Otherwise use the common 'gamma'
		public KernelDebugger(ProjKernel kernelType)
		{
			this(kernelType, KernelApproxTest.gamma);
		}
		
		public abstract double computeExact(FloatMat origCol1, FloatMat origCol2);
		
		public double test(boolean verbose)
		{
			PP.pTitledSectionLine("Kernel test: " + this.kernelType);
			double sumVal = 0, sumErr = 0; // w.r.t. exact kernel calculation
			double exact, approx;
			
			int pairN = batchSize - 1;
			for (int i = 0; i < pairN; i++)
			{
				approx = GpuBlas.dot(
						newData.createColOffset(i), 
						newData.createColOffset(i+1));
				exact = computeExact(
								origData.createColOffset(i), 
								origData.createColOffset(i+1));

				sumVal += Math.abs(exact);
				sumErr += Math.abs(exact - approx);
				if (verbose) PP.p("Exact", exact, "Approx", approx);
			}
			
			PP.pTitledSectionLine("Error Report", "-", 10);
	        PP.setSep();
			PP.setPrecision(2); 

			double avgPercentErr = sumErr / sumVal * 100;
			PP.setScientific(true);
			PP.p("Average absolute error =", sumErr / pairN);
			PP.setScientific(false);
			PP.p("Average percent error =", avgPercentErr , "%\n");
			assertEquals(this.kernelType + " FAILS", avgPercentErr, 0, TOL);
			PP.setPrecision(); // reset printer options
			
			return avgPercentErr;
		}
	}
	
	// Global tmp holder: reused
	final FloatMat diffOrig = new FloatMat(origDim + 1, 1, false);
	
	@Test
//	@Ignore
	public void gaussianTest()
	{
		new KernelDebugger(ProjKernel.Gaussian)
		{
			@Override
			public double computeExact(FloatMat origCol1, FloatMat origCol2)
			{
				double norm = GpuBlas.norm(
						GpuBlas.add( origCol1, origCol2, diffOrig, 1, -1));
    			return Math.exp(- gamma * norm * norm);
			}
		}.test(false);
	}
	
	@Test
//	@Ignore
	public void laplacianTest()
	{
		new KernelDebugger(ProjKernel.Laplacian)
		{
			@Override
			public double computeExact(FloatMat origCol1, FloatMat origCol2)
			{
				double norm = GpuBlas.add( origCol1, origCol2, diffOrig, 1, -1).abs_sum();
    			return Math.exp(- gamma * norm);
			}
		}.test(false);
	}
	
	@Test
//	@Ignore
	public void cauchyTest()
	{
		new KernelDebugger(ProjKernel.Cauchy)
		{
			@Override
			//  prod(1./(1 + gamma * (x1 - x2).^2/2))
			public double computeExact(FloatMat origCol1, FloatMat origCol2)
			{
				return GpuBlas.add( origCol1, origCol2, diffOrig, 1, -1)
						.square()
						.reciprocal(gamma, 1, 1)
						.product();
			}
		}.test(false);
	}
	
	@Test
	@Ignore
	public void visualTest()
	{
		//		Initializer.resetRand(); Initializer.laplacianProjKernelIniter(.5).init(m); PP.p(m, "\n\n");
		//		Initializer.resetRand(); Initializer.cauchyIniter(.5).init(m); PP.p(m, "\n\n");
		//		
		//		Initializer.resetRand(); Initializer.cauchyProjKernelIniter(.5).init(m); PP.p(m, "\n\n");
		//		Initializer.resetRand(); Initializer.laplacianIniter(.5).init(m); PP.p(m, "\n\n");
		//		
		//		Initializer.resetRand(); Initializer.projKernelIniter(ProjKernel.Gaussian, .5).init(m); PP.p(m, "\n\n");
		//		Initializer.resetRand(); Initializer.gaussianIniter(.5).init(m); PP.p(m, "\n\n");

		FloatMat a = new FloatMat(27, 5);
		Initializer.resetRand();
		Initializer.mixProjKernelAggregIniter(
				new Initializer[] {
						Initializer.fillIniter(10),
						Initializer.fillIniter(20),
						Initializer.fillIniter(30),
						Initializer.fillIniter(40)
				}, 
				4, 2, 4, 3)
				.init(a);
		PP.p(a);
	}
}