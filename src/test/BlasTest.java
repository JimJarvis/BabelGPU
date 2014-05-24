package test;

import static gpu.GpuBlas.*;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.*;
import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import static jcuda.Sizeof.*;
import gpu.FloatMat;
import gpu.GpuBlas;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.JCuda;
import utils.*;
import static utils.GpuUtil.*;
import static gpu.FloatMat.*;

public class BlasTest
{
	public static void main(String[] args)
	{
		JCuda.setExceptionsEnabled(true);
		JCurand.setExceptionsEnabled(true);

		Timer timer = Timer.getInstance();
		
		timer.start();
		
		/*
		 * Step 1: 7m x 360  (X)  360 x 10,000 + b
		 */
		float A[][] = new float[][] {{1, 5},
                                				{2, 6},
                                				{3, 7},
                                				{4, 8}};
		float B[][] = new float[][] {{-3, 0, -1},
                                				{-2, -5, 4}};


		// Create a CUBLAS handle
		GpuBlas.init();
		
		int m = A.length;
		int k = A[0].length;
		int n = B[0].length;

		// Allocate memory on the device
		FloatMat matA = new FloatMat(A);
		FloatMat matB = new FloatMat(B);
		
		// **************************************
		PP.pSectionLine();
		PP.p("Matrix-matrix Multiplication");
		// A * B
		FloatMat ansAB = GpuBlas.mult(matA, matB, 1);
		PP.po(ansAB);
		
		// 2 * A * B + (A*B)
		FloatMat ans3AB = new FloatMat(m, n);
		GpuBlas.mult(matA, matB, ans3AB);
		GpuBlas.mult(matA, matB, ans3AB, 2, 1);
		PP.po(ans3AB);
		
		// T = B' * A'
		FloatMat ansT = GpuBlas.mult(matB.transpose(), matA.transpose());
		PP.po(ansT);
		ansT.destroy();

		ansT = new FloatMat(m, m);
		GpuBlas.mult(matA, matA.transpose(), ansT, 0.5f, 0);
		PP.po(ansT);
		ansT.destroy();
		
		ansT = new FloatMat(n, n);
		GpuBlas.mult(matB.transpose(), matB, ansT, 1f, 0);
		PP.po(ansT);
		
		// **************************************
		PP.pSectionLine();
		PP.p("Matrix Addition");
		
        float C[][] = new float[][] {{3, 0},
                                				{-5, -1},
                                				{10, 4},
                                				{-9, -5}};
        FloatMat matC = new FloatMat(C);
        
        // A + C
        FloatMat ansApC = GpuBlas.add(matA, matC);
        PP.po(ansApC);
        
        // 2 * A + (-1) C
        FloatMat ans2AmC = new FloatMat(m, k, false);
        GpuBlas.add(matA, matC, ans2AmC, 2, -1);
        PP.po(ans2AmC);
        
        // C' + A'
        FloatMat ansCtpAt = new FloatMat(k, m, false);
        GpuBlas.add(matC.transpose(), matA.transpose(), ansCtpAt);
        PP.po(ansCtpAt);
        
		// **************************************
        PP.pSectionLine();
        PP.p("Matrix-vector multiplication");
        float Va[] = new float[] {10, 20};
        float Vb[] = new float[] {-5, -10, 0, 3};
        
        FloatMat vecA = new FloatMat(Va);
        FloatMat vecB = new FloatMat(Vb);
        
        FloatMat ansAvA = GpuBlas.multVec(matA, vecA);
        PP.po(ansAvA.transpose());
        
        FloatMat ans3AtvB = new FloatMat(2, 1);
        GpuBlas.multVec(matA.transpose(), vecB, ans3AtvB, 3, 100);
        PP.po(ans3AtvB.transpose());

        FloatMat ans1CtvB = GpuBlas.multVec(matC.transpose(), vecB, -1, 2);
        PP.po(ans1CtvB.transpose());
        
		// **************************************
        PP.pSectionLine();
        PP.p("Copy");
        FloatMat ansCopy = GpuBlas.copy(matA);
        PP.po(ansCopy);
        
		// **************************************
        PP.pSectionLine();
        PP.p("Max/Min absolute value index");
        FloatMat matM = new FloatMat(
        		new float[][] {{1, 3, 0},
                              		  {2, 5, 7},
                              		  {3, -5, 10},
                              		  {6, -100, 10}});

        int maxIdx = GpuBlas.maxAbsIndex(matM);
        PP.p("Max:", maxIdx);
        PP.p(matM.toCoord(maxIdx));
        PP.p(matM.toIndex(matM.toCoord(maxIdx))); // = maxIdx

        int minIdx = GpuBlas.minAbsIndex(matM);
        PP.p("Min:", minIdx);
        PP.p(matM.toCoord(minIdx));
        PP.p(matM.toIndex(matM.toCoord(minIdx))); // = maxIdx
        
		// **************************************
        PP.pSectionLine();
        PP.p("L2-Norm");
        FloatMat vecC = new FloatMat(new float[] {1, 2, -3, 4});
        PP.p(GpuBlas.norm(vecC));
        
		// **************************************
        PP.pSectionLine();
        PP.p("Scale and Add");
        FloatMat ansvBpvC = new FloatMat(4, 1);
        // -5, -10, 0, 3
        PP.p(GpuBlas.scaleAdd(vecB, ansvBpvC, 6).transpose());
        PP.p(GpuBlas.scaleAdd(vecC, ansvBpvC, -1).transpose());
        
        // Scale itself
        PP.p(GpuBlas.scale(ansvBpvC, 0.5f).transpose());
        
     // **************************************
        PP.pSectionLine();
        PP.p("Dot product");
        PP.p(GpuBlas.dot(vecC, vecB));

     // **************************************
        PP.pSectionLine();
        PP.p("Swap");
        GpuBlas.swap(vecB, vecC);
        PP.p("vecB:", vecB.transpose(), "and vecC:", vecC.transpose());
        GpuBlas.swap(vecB, vecC);
        PP.p("Swap back!\nvecB:", vecB.transpose(), "and vecC:", vecC.transpose());

		// Clean up
		FloatMat[] mats = new FloatMat[] 
				{matA, matB, ansAB, ans3AB, ansT, 
				  matC, ansApC, ans2AmC, ansCtpAt,
				  vecA, vecB, ansAvA, ans3AtvB, ans1CtvB, 
				  ansCopy,
				  matM,
				  vecC,
				  ansvBpvC};
		for (FloatMat mat : mats)
			mat.destroy();
		GpuBlas.destroy();
	}
}
