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
		FloatMat matA = new FloatMat(flatten(A), m, k);
		FloatMat matB = new FloatMat(flatten(B), k, n);
		
		FloatMat matC = GpuBlas.mult(matA, matB, 2);
		PP.po(deflatten(matC.getHost(), m));
		
		FloatMat matD = new FloatMat(m, n);
		GpuBlas.mult(matA, matB, matD);
		GpuBlas.mult(matA, matB, matD, 2, 1);
		PP.po(deflatten(matD.getHost(), m));

		// Clean up
		matA.destroy();
		matB.destroy();
		matC.destroy();
		matD.destroy();
		GpuBlas.destroy();
	}
}
