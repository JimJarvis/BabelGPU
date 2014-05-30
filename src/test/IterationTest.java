package test;

import utils.GpuUtil;
import utils.Timer;
import gpu.FloatMat;
import gpu.GpuBlas;
import gpu.GpuRand;
import jcuda.jcurand.JCurand;
import jcuda.runtime.JCuda;

/**
 * Test "iteration overhead"
 */
public class IterationTest
{
	
	//  A*B, r = rowDim(A), c = colDim(A), c2 = colDim(B)
	private static void testRandMat(Timer timer, int r, int c, int c2)
	{
		GpuRand grand = new GpuRand(111);
		FloatMat X = grand.genUniformFloat(r, c);
		FloatMat W = grand.genUniformFloat(c, c2);
		timer.readFromLast("Created random X and W");

		GpuBlas.mult(X, W);
		GpuUtil.synchronize();
		timer.readFromLast("X * W one shot");

		// Iteration column by column
		FloatMat res = new FloatMat(r, 1, false);
		FloatMat coltmp = new FloatMat();
		for (int i = 0; i < c2; i++)
		{
			W.createOffset(coltmp, c*i, c);
			GpuBlas.mult(X, coltmp, res);
		}
		GpuUtil.synchronize();
		timer.readFromLast("X * W column by column");
		
		X.destroy();
		W.destroy();
		grand.destroy();
	}

	public static void main(String[] args)
	{
		JCuda.setExceptionsEnabled(true);
        JCurand.setExceptionsEnabled(true);
        
        GpuBlas.init();
        
        Timer timer = Timer.getInstance();
        timer.start();
        
        int r = 25000;
        int c = 361;
        int c2 = 10001;
        
        testRandMat(timer, r, c, c2);
       
        r = 1000;
        c = 10000;
        c2 = 25000;
        
        testRandMat(timer, r, c, c2);
        
        GpuBlas.destroy();
	}

}
