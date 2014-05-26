package test;

import gpu.FloatMat;
import gpu.GpuBlas;
import gpu.Thrust;
import gpu.ThrustNative;
import utils.*;

public class BabelTest
{

	public static void main(String[] args)
	{
		// Initialize timer
		Timer timer = Timer.getInstance();
		timer.start();
		
		// Read in dummy data
		CsvReader csv = new CsvReader("test_X.txt");
		FloatMat X = new FloatMat(csv.readFloatMat());
		csv = new CsvReader("test_W.txt");
		FloatMat W = new FloatMat(csv.readFloatMat());
		csv = new CsvReader("test_Y.txt");
		int[] Y = csv.readIntVec(true);
		timer.readFromLast("Read the database from CSV");
		
		
		PP.p("X", X.row, X.col);
		PP.p("W", W.row, W.col);
		PP.p("Y", Y.length);
		PP.setDoublePrec(1);
		
		/*
		 * Dimensions
		 */
		csv = new CsvReader("test_dim.txt");
		int[] dims = csv.readIntVec();
		final int SAMPLES = dims[0];
		final int X_DIM = dims[1];
		final int X_NEW_DIM = dims[2];
		final int LABELS = dims[3];
		
		/*
		 * Define a few learning constants
		 */
		final float LearningRate = 1.5f;
		final float Lambda = 2f;
		
		GpuBlas.init();
		timer.readFromLast("cuBLAS init");

		/*
		 * Step 1: cos(W * x + b)
		 * W and b are combined. 
		 */
		// augment X with a column of 1
		FloatMat X1 = new FloatMat(SAMPLES, X_DIM + 1);
		X1.copyFrom(X);
		ThrustNative.gpu_fill_float(X1.getThrustPointer().offset(X.size()), SAMPLES, 1);
//		X1.getHostFromDevice(); PP.p(X1);
		
		// Xnew: LABELS * SAMPLES
		FloatMat Xnew = GpuBlas.mult(W, X1.transpose()).cos();
		timer.readFromLast("Step 1");
		
		/*
		 * Step2: Create Theta matrix and compute Theta * X_new
		 */
		FloatMat Theta = new FloatMat(LABELS, X_NEW_DIM);

		FloatMat A = new FloatMat(LABELS, 1, false);
		// Loop over samples column by column
		for (int s = 0; s < SAMPLES; ++ s)
		{
			// Step2: extract a column
			FloatMat Xnew_s = Xnew.createOffset(s * X_NEW_DIM, X_NEW_DIM);
			// alpha_vector = Theta * Xnew_s, LABELS * 1
			GpuBlas.mult(Theta, Xnew_s, A);
			
			/*
			 * Step3: get Id[y==j] - P(yj | x, Theta)
			 */
			Thrust.babel_id_minus_softmax(A, Y[s]);
			
			// Step3: update Theta
			// Theta += Lr * ( (Id-P) * Xnew_s' - Lambda/SAMPLES * Theta)
			GpuBlas.mult(A, Xnew_s.transpose(), Theta, 
					LearningRate, 1 - LearningRate * Lambda / SAMPLES);
			
			timer.readFromLast("Iteration " + s);
		}

		// DONE!
		PP.p(Theta);
		
		/*
		 * Clean up and exit
		 */
		FloatMat[] mats = new FloatMat[] 
				{X, W, X1, Xnew};
		for (FloatMat mat : mats)
			mat.destroy();
		GpuBlas.destroy();
	}

}
