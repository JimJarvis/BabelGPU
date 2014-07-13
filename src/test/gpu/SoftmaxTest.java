package test.gpu;

import utils.*;
import gpu.*;

import com.googlecode.javacpp.*;

public class SoftmaxTest
{
	private static final float TOL = 1e-9f;
	
	private static int ROW;
	private static int COL;
	public static void main(String[] args)
	{
		GpuUtil.enableExceptions();
		
		PP.p("Softmax test");
		PP.setPrecision(3);
		
		/*
		 * Dimensions
		 */
		CsvReader csv = new CsvReader("matlab_test/Softmax_input_dim.txt");
		int[] dims = csv.readIntVec(true);
		ROW = dims[0]; COL = dims[1];
		
		// Read in dummy data
		FloatMat X = rereadX();
		csv = new CsvReader("input_Y.txt");
		int[] labels = csv.readIntVec(true);
		IntPointer labelsDevice = Thrust.copy_host_to_device(labels);
		
		/*
		 * softmax(X) in full
		 */
		Thrust.batch_softmax(X);
		GpuUtil.checkGold(X, "gold_softmax", "softmax(full)", TOL);
		X.destroy();
		
		/**
		 * softmax(X) return only the probability at the correct label of each column
		 */
		X = rereadX();
		FloatMat maxProbs = new FloatMat(1, COL, false);
		Thrust.batch_softmax_at_label(X, maxProbs, labelsDevice);
		GpuUtil.checkGold(maxProbs, "gold_softmax_labeled", "softmax(correct label)", TOL);
		
		// compute sum of log likelihood
		float logProb = Thrust.log_sum(maxProbs);
		float goldLogProb = new CsvReader("gold_log_prob.txt").readFloatVec(true)[0];
		
		PP.p("[log likelihood] " + 
				(Math.abs(goldLogProb - logProb) < TOL ? "PASS" : "FAIL"));
		
		X.destroy();	maxProbs.destroy();

		/**
		 * Label where the maximum probability occurs
		 */
		X = rereadX();
		IntPointer reusedPtr = Thrust.malloc_device_int(COL);
		final int dummyOffset = 766;
		int outLabels[] = new int[COL + dummyOffset]; // 66 dummy offset
		Thrust.best_label(X, reusedPtr, outLabels, dummyOffset);
		int[] goldLabels = new CsvReader("gold_best_labels.txt").readIntVec(true);
		// checkGold
		int fault = -1; // if stays -1, then test passes
		for (int i = 0; i < COL; i ++)
			if (outLabels[i + dummyOffset] != goldLabels[i])
			{
				fault = i;
				break;
			}
		if (fault != -1)
			PP.p("[best label]: FAIL", fault);
		else
			PP.p("[best label]: PASS");
			
		
		Thrust.free_device(labelsDevice);
		Thrust.free_device(reusedPtr);
		X.destroy();;
	}
	
	private static FloatMat rereadX()
	{
		CsvReader csv = new CsvReader("input_X.txt");
		return new FloatMat(csv.readFloatVec(true), ROW, COL);
	}
}
