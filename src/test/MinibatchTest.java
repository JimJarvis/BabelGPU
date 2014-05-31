package test;

import utils.PP;

import com.googlecode.javacpp.IntPointer;
import com.googlecode.javacpp.Loader;

import gpu.FloatMat;
import gpu.Thrust;

public class MinibatchTest
{
	public static void main(String[] args)
	{
		FloatMat mat = new FloatMat(new float[] 
				{ 4.2f, 5.9f, -2.1f, -3.7f, 3.3f, 1.9f, -0.6f, 2.5f, 1.7f, -0.2f, -0.9f, 0.4f}, 4, 3);
		int labels[] = new int[] {0, 3, 2, 1};
		
		IntPointer labelsDevice = Thrust.copy_host_to_device(labels);
		Thrust.babel_batch_id_minus_softmax_float(mat, labelsDevice);
		
		Thrust.gpu_free(labelsDevice);
		
		PP.p(mat);
	}

}
