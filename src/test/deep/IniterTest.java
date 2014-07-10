package test.deep;

import java.util.ArrayList;
import java.util.Arrays;

import utils.PP;
import deep.Initializer;
import deep.Initializer.ProjKernel;
import gpu.FloatMat;

public class IniterTest
{
	public static void main(String[] args)
	{
		FloatMat m = new FloatMat(7, 5);
		
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
		ArrayList<Initializer> initers = new ArrayList<>();
//		initers.add(Initializer.laplacianIniter(.5));
//		initers.add(Initializer.gaussianIniter(.5));
//		initers.add(Initializer.cauchyIniter(.5));
//		initers.add(Initializer.gaussianIniter(.5));
		initers.add(Initializer.fillIniter(10));
		initers.add(Initializer.fillIniter(20));
		initers.add(Initializer.fillIniter(30));
		initers.add(Initializer.fillIniter(40));
		ArrayList<Double> ratios = new ArrayList<>();
		ratios.add(4.);
		ratios.add(2.);
		ratios.add(4.);
		ratios.add(3.);
		Initializer.mixProjKernelAggregIniter(initers, ratios).init(a);
		PP.p(a);
	}
}
