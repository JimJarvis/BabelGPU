package test;

import gpu.GpuException;
import gpu.GpuRand;
import jcuda.Pointer;
import jcuda.jcurand.JCurand;
import static jcuda.runtime.JCuda.*;
import utils.PP;
import utils.Timer;

public class RandTest
{

	public static void main(String[] args) throws GpuException
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
		rand.genNormalFloat(20, 0, 3);
		timer.readFromLast();
		PP.p(rand.getHostArray());
		timer.readFromLast();
		
		PP.p("Uniform");
		// If you explicitly obtain the pointer, you must
		// manually free it
		rand.genUniformFloat(20);
		Pointer device = rand.getDevice();
		timer.readFromLast();
		PP.p(rand.getHostArray());
		cudaFree(device);
		timer.readFromLast();
		
		PP.p("Normal 2^26 floats");
		rand.genNormalFloat(1 << 27, 0, 3);
		timer.readFromLast();
		rand.getHostArray();
		timer.readFromLast();
		
		rand.destroy();
	}

}
