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
		
		PP.pSectionLine();
		PP.p("Matrix-matrix Multiplication");
		// A * B
		FloatMat ansAB = GpuBlas.mult(matA, matB, 1);
		PP.po(ansAB.deflatten());
		
		// 2 * A * B + (A*B)
		FloatMat ans3AB = new FloatMat(m, n);
		GpuBlas.mult(matA, matB, ans3AB);
		GpuBlas.mult(matA, matB, ans3AB, 2, 1);
		PP.po(ans3AB.deflatten());
		
		// T = B' * A'
		FloatMat ansT = GpuBlas.mult(matB.transpose(), matA.transpose());
		PP.po(ansT.deflatten());
		ansT.destroy();

		ansT = new FloatMat(m, m);
		GpuBlas.mult(matA, matA.transpose(), ansT, 0.5f, 0);
		PP.po(ansT.deflatten());
		ansT.destroy();
		
		ansT = new FloatMat(n, n);
		GpuBlas.mult(matB.transpose(), matB, ansT, 1f, 0);
		PP.po(ansT.deflatten());
		
		PP.pSectionLine();
		PP.p("Matrix Addition");
		
        float C[][] = new float[][] {{3, 0},
                                				{-5, -1},
                                				{10, 4},
                                				{-9, -5}};
        FloatMat matC = new FloatMat(C);
        
        // A + C
        FloatMat ansApC = GpuBlas.add(matA, matC);
        PP.po(ansApC.deflatten());
        
        // 2 * A + (-1) C
        FloatMat ans2AmC = new FloatMat(m, k);
        GpuBlas.add(matA, matC, ans2AmC, 2, -1);
        PP.po(ans2AmC.deflatten());
        
        // C' + A'
        FloatMat ansCtpAt = new FloatMat(k, m);
        GpuBlas.add(matC.transpose(), matA.transpose(), ansCtpAt);
        PP.po(ansCtpAt.deflatten());
        
        PP.pSectionLine();
        PP.p("Matrix-vector multiplication");
        float Va[] = new float[] {10, 20};
        float Vb[] = new float[] {-5, -10, 0, 3};
        
        FloatMat vecA = new FloatMat(Va);
        FloatMat vecB = new FloatMat(Vb);
        
        FloatMat ansAvA = GpuBlas.multVec(matA, vecA);
        PP.po(ansAvA.transpose().deflatten());
        
        FloatMat ans3AtvB = new FloatMat(2, 1);
        GpuBlas.multVec(matA.transpose(), vecB, ans3AtvB, 3, 100);
        PP.po(ans3AtvB.transpose().deflatten());

        FloatMat ans1CtvB = GpuBlas.multVec(matC.transpose(), vecB, -1, 2);
        PP.po(ans1CtvB.transpose().deflatten());
        

		// Clean up
		FloatMat[] mats = new FloatMat[] 
				{matA, matB, ansAB, ans3AB, ansT, 
				  matC, ansApC, ans2AmC, ansCtpAt,
				  vecA, vecB, ansAvA, ans3AtvB, ans1CtvB};
		for (FloatMat mat : mats)
			mat.destroy();
		GpuBlas.destroy();
	}
}
