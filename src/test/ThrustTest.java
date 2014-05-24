package test;

import utils.PP;
import gpu.FloatMat;
import gpu.Thrust;
import gpu.Thrust.*;

public class ThrustTest
{

	public static void main(String[] args)
	{
		FloatMat vec = new FloatMat(new float[] {4, 3, 2, 1, 10, -5});
		FloatMat out = new FloatMat(6, 1);
		FloatDevicePointer fp = new FloatDevicePointer(vec.getDevice());
		FloatDevicePointer fplast = new FloatDevicePointer(fp.get().position(vec.size()));
		FloatDevicePointer fout = new FloatDevicePointer(out.getDevice());
		Thrust.sort(fp, fplast);
//		Thrust.transform(fp, fplast, fout, Thrust.exp);
		
		float ans = Thrust.reduce(fp, fplast, 0, new FloatPlus());
		
		PP.p(vec.getHost());
		PP.p(vec.getHostFromDevice());
		PP.p("ans:", ans);
	}

}
