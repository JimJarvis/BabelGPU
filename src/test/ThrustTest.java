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
		FloatDevicePointer fp = new FloatDevicePointer(vec.getDevice());
		FloatDevicePointer fplast = new FloatDevicePointer(fp.get().position(vec.size()));
		Thrust.sort(fp, fplast);
		float ans = Thrust.reduce(fp, fplast, 0, new FloatPlus());
		
		PP.p("Forking ans:", ans);
	}

}
