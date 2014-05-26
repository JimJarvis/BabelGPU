package test;

import gpu.FloatMat;
import gpu.GpuBlas;
import gpu.ThrustNative;
import utils.CsvReader;
import utils.PP;

public class BabelTest
{

	public static void main(String[] args)
	{
		// Read in dummy data
		CsvReader csv = new CsvReader("dummy/test_X.txt");
		FloatMat X = new FloatMat(csv.readFloatMat());
		csv = new CsvReader("dummy/test_W.txt");
		FloatMat W = new FloatMat(csv.readFloatMat());
		csv = new CsvReader("dummy/test_Y.txt");
		FloatMat Y = new FloatMat(csv.readFloatMat());
		
		PP.p("X", X.row, X.col);
		PP.p("W", W.row, W.col);
		PP.p("Y", Y.row, Y.col);
		PP.setDoublePrec(1);
		
		/*
		 * Dimensions
		 */
		csv = new CsvReader("dummy/test_dim.txt");
		int[] dims = csv.readIntVec();
		final int SAMPLES = dims[0];
		final int X_DIM = dims[1];
		final int X_NEW_DIM = dims[2];
		final int LABELS = dims[3];
		
		GpuBlas.init();

		/*
		 * Step 1: cos(W * x + b)
		 * W and b are combined. 
		 */
		// augment X with a column of 1
		FloatMat X1 = new FloatMat(SAMPLES, X_DIM + 1);
		X1.copyFrom(X);
		ThrustNative.gpu_fill_float(X1.getThrustPointer().offset(X.size()), SAMPLES, 1);
//		X1.getHostFromDevice(); PP.p(X1);
		
		FloatMat X_new = GpuBlas.mult(W, X1.transpose()).cos();
		
		/*
		 * Step2: Create Theta matrix and compute Theta * X_new
		 */
		FloatMat Theta = new FloatMat(LABELS, X_NEW_DIM);
		
		// Loop over samples column by column
		for (int s = 0; s < SAMPLES; ++ s)
		{
			// Step 2
		}

		
		/*
		 * Clean up and exit
		 */
		FloatMat[] mats = new FloatMat[] 
				{X, W, X1, X_new};
		for (FloatMat mat : mats)
			mat.destroy();
		GpuBlas.destroy();
	}

}
