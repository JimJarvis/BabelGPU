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
		FloatDevicePointer fplast = new FloatDevicePointer(fp.get().position(vec.size()));
		ThrustStruct.sort(fp, fplast);
		
//		FloatPointer fptr = new FloatPointer(FloatBuffer.wrap(f));
//		Thrust.transform(fp, fplast, fout, Thrust.exp);
		
		float ans = ThrustStruct.reduce(fp, fplast, 0, new FloatPlus());
		
//		FloatMat mx = new FloatMat( jcuda.Pointer.to(d.get().asBuffer()),1,1);
		
		
		PP.p(vec.getHost());
		PP.p(vec.getHostFromDevice());
		PP.p("ans:", ans);
		PP.p("max:", Thrust.gpu_max_float(fp, fplast));
		
		Thrust.gpu_exp_float(fp, fplast, 0.5f, 3);
		PP.p(vec.getHostFromDevice());
		
		PP.p(Thrust.babel_id_minus_softmax(fp, fplast, 0) + 10);
	}

}
