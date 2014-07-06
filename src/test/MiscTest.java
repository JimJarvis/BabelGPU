package test;

import deep.units.*;
import gpu.*;
import utils.*;

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
		
		FloatMat b = new FloatMat(B.clone());
		FloatMat a = new FloatMat(A.clone());
		
		b.singleIncr(2, 1, 666);
		PP.p(b);
		a.singleSet(2, 0, 666);
		PP.p(a);
		
		PP.p(GpuBlas.dotMult(a, b));
		PP.p(b.reciprocal());
		PP.p(b.reciprocal().square());
		b = new FloatMat(B.clone());
		PP.p(b.cube());
		FloatMat c = new FloatMat(6, 1, false);
		PP.p(GpuBlas.dotMult(a, b, c));
		a = new FloatMat(A);
		PP.p(a.sigmoid_deriv());
		
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
		
		PP.p( GpuUtil.getGpuInfo());
	}
}
