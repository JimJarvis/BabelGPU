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
		
		// A * B
		FloatMat matAB = GpuBlas.mult(matA, matB, 1);
		PP.po(matAB.deflatten());
		
		// 2 * A * B + (A*B)
		FloatMat mat3AB = new FloatMat(m, n);
		GpuBlas.mult(matA, matB, mat3AB);
		GpuBlas.mult(matA, matB, mat3AB, 2, 1);
		PP.po(mat3AB.deflatten());
		
		// T = B' * A'
		FloatMat matT = GpuBlas.mult(matB.transpose(), matA.transpose());
		PP.po(matT.deflatten());
		matT.destroy();

		matT = new FloatMat(m, m);
		GpuBlas.mult(matA, matA.transpose(), matT, 0.5f, 0);
		PP.po(matT.deflatten());
		matT.destroy();
		
		matT = new FloatMat(n, n);
		GpuBlas.mult(matB.transpose(), matB, matT, 1f, 0);
		PP.po(matT.deflatten());
		
		PP.pSectionLine();
		PP.p("Matrix Addition");
		
        float C[][] = new float[][] {{3, 0},
                                				{-5, -1},
                                				{10, 4},
                                				{-9, -5}};
        FloatMat matC = new FloatMat(C);
        
        FloatMat mat1 = GpuBlas.add(matA, matC);
        PP.po(mat1.deflatten());
        
        FloatMat mat2 = new FloatMat(m, k);
        GpuBlas.add(matA, matC, mat2, 2, -1);
        PP.po(mat2.deflatten());
        
        FloatMat mat3 = new FloatMat(k, m);
        GpuBlas.add(matC.transpose(), matA.transpose(), mat3);
        PP.po(mat3.deflatten());

		// Clean up
		FloatMat[] mats = new FloatMat[] {matA, matB, matAB, mat3AB, matC, matT, mat1, mat2, mat3};
		for (FloatMat mat : mats)
			mat.destroy();
		GpuBlas.destroy();
	}
}
