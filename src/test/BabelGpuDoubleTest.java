package test;

import jcuda.jcurand.JCurand;
import jcuda.runtime.JCuda;
import gpu.*;
import utils.*;

public class BabelGpuDoubleTest
{
	private static final double TOL = 1e-11;
	private static final int ITER = 1000;

	public static void main(String[] args)
	{
		JCuda.setExceptionsEnabled(true);
        JCurand.setExceptionsEnabled(true);
        
        PP.p("Double GPU test");
        
		// Initialize timer
		Timer timer = Timer.getInstance();
		timer.start();
		
		// Read in dummy data
		CsvReader csv = new CsvReader("input_X.txt");
		DoubleMat X = new DoubleMat(csv.readDoubleMat());
		csv = new CsvReader("input_W.txt");
		DoubleMat W = new DoubleMat(csv.readDoubleMat());
		csv = new CsvReader("input_Y.txt");
		int[] Y = csv.readIntVec(true);
		timer.readFromLast("Read the database from CSV");
		
		/*
		 * Dimensions
		 */
		csv = new CsvReader("input_dim.txt");
		int[] dims = csv.readIntVec(true);
		final int SAMPLES = dims[0];
		final int X_DIM = dims[1];
		final int X_NEW_DIM = dims[2];
		final int LABELS = dims[3];
		
		/*
		 * Define a few learning constants
		 */
		csv = new CsvReader("input_learn.txt");
		double[] learns = csv.readDoubleVec(true);
		final double LearningRate = learns[0];
		final double Lambda = learns[1];
		
		GpuBlas.init();
		timer.readFromLast("cuBLAS init");

		/*
		 * Step 1: cos(W * x + b)
		 * W and b are combined. 
		 */
		// augment X with a column of 1
		DoubleMat X1 = new DoubleMat(SAMPLES, X_DIM + 1, false);
		X1.copyFrom(X);
		ThrustNative.gpu_fill_double(X1.getThrustPointer().offset(X.size()), SAMPLES, 1);
		
//		X1.getHostFromDevice(); checkGold(X1.deflatten(), "X1");
		
//		DoubleMat WX = GpuBlas.mult(W, X1.transpose());
		
		// Xnew: X_NEW_DIM * SAMPLES
//		checkGold(GpuBlas.mult(W, X1.transpose()), "WX");
		
		DoubleMat Xnew = GpuBlas.mult(W, X1.transpose()).cos();
		timer.readFromLast("Step 1");
//		checkGold(Xnew, "Xnew");

		/*
		 * Step2: Create Theta matrix and compute Theta * X_new
		 */
		DoubleMat Theta = new DoubleMat(LABELS, X_NEW_DIM);

		DoubleMat A = new DoubleMat(LABELS, 1, false);
		// Loop over samples column by column
		for (int s = 0; s < ITER; ++ s)
		{
			// Step2: extract a column
			DoubleMat Xnew_s = Xnew.createOffset(s * X_NEW_DIM, X_NEW_DIM);
			// alpha_vector = Theta * Xnew_s, LABELS * 1
			GpuBlas.mult(Theta, Xnew_s, A);
			
			/*
			 * Step3: get Id[y==j] - P(yj | x, Theta)
			 */
			Thrust.babel_id_minus_softmax_double(A, Y[s]);
			
			// Step3: update Theta
			// Theta += Lr * ( (Id-P) * Xnew_s' - Lambda/SAMPLES * Theta)
			GpuBlas.mult(A, Xnew_s.transpose(), Theta, 
					LearningRate, 1 - LearningRate * Lambda / SAMPLES);
		}
//		checkGold(A, "A");

		/*
		 * DONE!
		 * Check results against Matlab
		 */
		PP.p("Done. Check Theta:");
		checkGold(Theta, "Theta");
		
		PP.p("Theta abs avg:", Theta.abs().sum() / Theta.size());

		/*
		 * Clean up and exit
		 */
		DoubleMat[] mats = new DoubleMat[] 
				{X, W, X1, Xnew, Theta};
		for (DoubleMat mat : mats)
			mat.destroy();
		GpuBlas.destroy();
	}
	
	/**
	 * Check the gold standard generated from Matlab
	 */
	private static void checkGold(DoubleMat gpu, String goldFile)
	{
		CsvReader csv = new CsvReader("gold_" + goldFile + ".txt");
		double[][] Gold = csv.readDoubleMat();
		double[][] Host = gpu.deflatten();
		
		double diff = GpuUtil.matAvgDiff(Gold, Host);
		PP.setDoublePrec(3);
		PP.setScientific(true);
		
		if (GpuUtil.matAvgDiff(Gold, Host) < TOL)
    		PP.p("PASS double GPU-Matlab: ", diff);
		else
			PP.p("FAIL double GPU-Matlab: ", diff);
	}

}
