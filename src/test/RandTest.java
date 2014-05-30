package test;

import gpu.FloatMat;
import gpu.GpuRand;
import jcuda.Pointer;
import jcuda.jcurand.JCurand;
import static jcuda.runtime.JCuda.*;
import utils.PP;
import utils.Timer;

public class RandTest
{

	public static void main(String[] args)
	{
		setExceptionsEnabled(true);
		JCurand.setExceptionsEnabled(true);

		Timer timer = Timer.getInstance();
		
		timer.start();
		
		// Creation with seed
		PP.p("Curand creation");
		GpuRand rand = new GpuRand(37);
		timer.readFromLast();
		
		PP.p("Normal");
		FloatMat m = rand.genNormalFloat(20, 0, 3);
		timer.readFromLast();
		PP.p(m);
		timer.readFromLast();
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
		m.getHostFromDevice();
		timer.readFromLast();
		
		m.destroy();
		rand.destroy();
	}

}
