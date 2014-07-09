package test.gpu;

import gpu.FloatMat;
import gpu.GpuRand;
import jcuda.Pointer;
import jcuda.jcurand.JCurand;
import static jcuda.runtime.JCuda.*;
import utils.GpuUtil;
import utils.PP;
import utils.Timer;

public class RandTest
{

	public static void main(String[] args)
	{
		GpuUtil.enableExceptions();

		Timer timer = Timer.getInstance();
		
		timer.start();
		
		// Creation with seed
		PP.p("Curand creation");
		GpuRand rand = new GpuRand(37);
		timer.readFromLast();
		
		PP.p("Normal");
		FloatMat m = rand.genNormalFloat(20, 0, 3);
		rand.genCauchyFloat(m);
		timer.readFromLast();
		PP.p(m);
		timer.readFromLast();
		rand.resetSeed(37);
		m = rand.genNormalFloat(20, 0, 3);
		PP.p(m);
		
		m.destroy();
		
		PP.p("Uniform");
		// If you explicitly obtain the pointer, you must
		// manually free it
		m = rand.genUniformFloat(20);
		timer.readFromLast();
		PP.p(m);
		timer.readFromLast();
		m.destroy();
		
		PP.p("Normal 2^26 floats");
		m = rand.genNormalFloat(1 << 27, 0, 3);
		timer.readFromLast();
		m.toHostArray();
		timer.readFromLast();
		
		m.destroy();
		
		rand.destroy();
	}

}
