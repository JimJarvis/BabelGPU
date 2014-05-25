package test;

import utils.PP;
import gpu.*;
import gpu.ThrustStruct.*;

public class ThrustTest
{

	public static void main(String[] args)
	{
		float[] f = new float[] {4, 3, 2, 1, 10, -5};
		FloatMat vec = new FloatMat(f);
		FloatMat out = new FloatMat(6, 1);
		FloatDevicePointer fp = new FloatDevicePointer(vec.getDevice());
//		FloatDevicePointer fplast = new FloatDevicePointer(fp.get().position(vec.size()));
		FloatDevicePointer fp1 = fp.offset(1);
		
		
		PP.p(vec.getHost());
		PP.p(vec.getHostFromDevice());
		
		PP.p(Thrust.gpu_sum_float(fp1, vec.size() -1));
		Thrust.gpu_exp_float(fp1, vec.size() - 1, 0.5f, 3);
		PP.p(vec.getHostFromDevice());
		
	}

}
