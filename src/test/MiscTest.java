package test;

import gpu.FloatMat;
import gpu.GpuBlas;
import gpu.Thrust;
import gpu.ThrustNative;
import utils.GpuUtil;
import utils.PP;

public class MiscTest
{
	public static void main(String[] args)
	{
		GpuBlas.init();
		float A[] = new float[] {1, 2, 3, 4, 5, 6};
		float B[][] = new float[][] {{1, 10, 6},
												{2, 20, -2},
												{3, 30, -7},
												{4, 40, -10}};
		
		PP.po(FloatMat.deflatten(A, 3));
		PP.po(FloatMat.flatten(B));
		
		FloatMat m = new FloatMat(4, 3);
		PP.p(m.toCoord(10));
		PP.p(m.toIndex(3, 2));
		
		FloatMat b = new FloatMat(B);
		FloatMat a = new FloatMat(A);
		
		PP.setSep("\n");
		Thrust.set_last_row_one(b);
		Thrust.fill_col(b, -2, 1000);
		Thrust.fill_row(b, -2, -30);
		PP.p((Object[]) b.deflatten());
		
		b.destroy();
		b = new FloatMat(B);
		Thrust.fill_row(b, 2, 30);
		Thrust.fill_col(b, 1, -1000);
		PP.p((Object[]) b.deflatten());
		
		GpuBlas.destroy();
		
//		PP.p( GpuUtil.getGpuInfo());
	}
}
