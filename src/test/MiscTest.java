package test;

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
		
		GpuRand rand = new GpuRand(10);
		FloatMat dumd = rand.genUniformFloat(10000000);
		Timer timer = Timer.getInstance();
		Timer.setPrecision(4);
		timer.start();
		float[] dumh = dumd.getHostFromDevice();
		GpuUtil.synchronize();
		timer.readFromLast("Copy to CPU");
		dumd.destroy();
		timer.start();
		double[] dumh_ = new double[dumh.length];
		for (int i = 0 ; i < dumh.length; i ++)
			dumh_[i] = (double) dumh[i];
		timer.readFromLast("Casting to double");
		timer.start();
		for (int i = 0 ; i < dumh.length; i ++)
			dumh[i] = (float) dumh_[i];
		timer.readFromLast("Casting to float");
		FloatMat dumd_ = new FloatMat(dumh);
		timer.start();
		dumd_.getDeviceFromHost();
		GpuUtil.synchronize();
		timer.readFromLast("Copy to GPU");
	}
}
