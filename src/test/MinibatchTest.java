package test;

import utils.*;
import gpu.*;

import com.googlecode.javacpp.*;

public class MinibatchTest
{
	private static final float TOL = 1e-5f;
	
	public static void main(String[] args)
	{
		GpuUtil.enableExceptions();
		
		PP.p("Mini-batch test");
		PP.setPrecision(3);
		
		/*
		 * Dimensions
		 */
		CsvReader csv = new CsvReader("input_dim.txt");
		int[] dims = csv.readIntVec(true);
		final int ROW = dims[0];
		final int COL = dims[1];
		
		// Read in dummy data
		csv = new CsvReader("input_X.txt");
		FloatMat X = new FloatMat(csv.readFloatVec(true), ROW, COL);
		csv = new CsvReader("input_Y.txt");
		int[] labels = csv.readIntVec(true);
		
		IntPointer labelsDevice = Thrust.copy_host_to_device(labels);
		Thrust.babel_batch_id_minus_softmax_float(X, labelsDevice);
		
		GpuUtil.checkGold(X, "gold_MB", "Mini-batch", TOL);
		
		Thrust.gpu_free(labelsDevice);
	}

}
