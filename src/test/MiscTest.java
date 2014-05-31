package test;

import gpu.FloatMat;
import gpu.GpuBlas;
import utils.GpuUtil;
import utils.PP;

public class MiscTest
{
	public static void main(String[] args)
	{
		GpuBlas.init();
		float A[] = new float[] {1, 2, 3, 4, 5, 6};
		float B[][] = new float[][] {{1, 10},
												{2, 20},
												{3, 30},
												{4, 40}};
		
		PP.po(FloatMat.deflatten(A, 3));
		PP.po(FloatMat.flatten(B));
		
		FloatMat m = new FloatMat(4, 3);
		PP.p(m.toCoord(10));
		PP.p(m.toIndex(3, 2));
		
		FloatMat b = new FloatMat(B);
		FloatMat a = new FloatMat(A);
		GpuBlas.destroy();
		
		PP.p( GpuUtil.getGpuInfo());
	}
}
