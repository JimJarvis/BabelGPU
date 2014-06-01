package test;

import java.util.Arrays;

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
		int[] labelstmp = csv.readIntVec(true);
        // add 3 garbages values at the beginning to test Thrust.offset()
		int[] labels = new int[labelstmp.length + 3];  
		labels[0] = -1000; labels[1] = -2000; labels[2] = -3000;
		for (int i = 0; i < labelstmp.length; i ++)
			labels[i + 3] = labelstmp[i];
		
		IntPointer labelsDevice = Thrust.copy_host_to_device(labels);
		labelsDevice = Thrust.offset(labelsDevice, 3);
		Thrust.babel_batch_id_minus_softmax_float(X, labelsDevice);
		
		GpuUtil.checkGold(X, "gold_MB", "Mini-batch", TOL);
		
		Thrust.gpu_free(labelsDevice);
	}

}
