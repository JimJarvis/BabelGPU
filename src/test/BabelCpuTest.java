package test;

import gpu.FloatMat;
import gpu.GpuBlas;
import gpu.Thrust;
import gpu.ThrustNative;
import utils.*;

public class BabelCpuTest
{
	private static final float TOL = 1e-1f;
	
	public static void main(String[] args)
	{
		// Read in dummy data
		CsvReader csv = new CsvReader("input_X.txt");
		float[][] jX = csv.readFloatMat();
		FloatMat X = new FloatMat(jX);
		csv = new CsvReader("input_W.txt");
		
		float[][] jW = csv.readFloatMat();
		FloatMat W = new FloatMat(jW);
		csv = new CsvReader("input_Y.txt");
		int[] Y = csv.readIntVec(true);
		
		
		PP.p("X", X.row, X.col);
		PP.p("W", W.row, W.col);
		PP.p("Y", Y.length);
		PP.setDoublePrec(1);
		
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
		float[] learns = csv.readFloatVec(true);
		final float LearningRate = learns[0];
		final float Lambda = learns[1];
		
		GpuBlas.init();

		/*
		 * Step 1: cos(W * x + b)
		 * W and b are combined. 
		 */
		// augment X with a column of 1
		FloatMat X1 = new FloatMat(SAMPLES, X_DIM + 1, false);
		X1.copyFrom(X);
		ThrustNative.gpu_fill_float(X1.getThrustPointer().offset(X.size()), SAMPLES, 1);
		float[][] jX1 = addCol1(jX);
		
		// Xnew: X_NEW_DIM * SAMPLES
		FloatMat Xnew = GpuBlas.mult(W, X1.transpose()).cos();
		float[][] jXnew = cos(mult(jW, transpose(jX1)));
		PP.p("check Xnew"); 
		checkGold(Xnew, jXnew);
		checkGold(jXnew, "Xnew");

		/*
		 * Step2: Create Theta matrix and compute Theta * X_new
		 */
		FloatMat Theta = new FloatMat(LABELS, X_NEW_DIM);
		float[][] jTheta = new float[LABELS][X_NEW_DIM];

		FloatMat A = new FloatMat(LABELS, 1, false);
		float[][] jA = new float[LABELS][1];
		// Loop over samples column by column
		for (int s = 0; s < SAMPLES; ++ s)
		{
			// Step2: extract a column
			FloatMat Xnew_s = Xnew.createOffset(s * X_NEW_DIM, X_NEW_DIM);
			float[][] jXnew_s = getCol(jXnew, s);
			

			// alpha_vector = Theta * Xnew_s, LABELS * 1
			GpuBlas.mult(Theta, Xnew_s, A);
			jA = mult(jTheta, jXnew_s);
			
			/*
			 * Step3: get Id[y==j] - P(yj | x, Theta)
			 */
			Thrust.babel_id_minus_softmax(A, Y[s]);
			jbabel_id_minus_softmax(jA, Y[s]);
			
			// Step3: update Theta
			// Theta += Lr * ( (Id-P) * Xnew_s' - Lambda/SAMPLES * Theta)
			GpuBlas.mult(A, Xnew_s.transpose(), Theta, 
					LearningRate, 1 - LearningRate * Lambda / SAMPLES);
			
			updateTheta(jTheta, - LearningRate * Lambda / SAMPLES, mult(jA, transpose(jXnew_s)), LearningRate);

//		PP.p("Iteration", s);
//		checkGold(A, jA);
//		checkGold(Theta, jTheta);
		}
		PP.p("Check vector A");
		checkGold(A, jA);
		checkGold(jA, "A");

		/*
		 * DONE!
		 * Check results against plain Java
		 */
		PP.p("Done. Check Theta:");
		checkGold(Theta, jTheta);
		checkGold(jTheta, "Theta");

		/*
		 * Clean up and exit
		 */
		FloatMat[] mats = new FloatMat[] 
				{X, W, X1, Xnew};
		for (FloatMat mat : mats)
			mat.destroy();
		GpuBlas.destroy();
	}
	
	/**
	 * Check the gold standard from plain Java
	 */
	private static void checkGold(FloatMat hostMat, float[][] Gold)
	{
		float[][] Host = hostMat.deflatten();
		
		for (int i = 0; i < Gold.length; i ++)
			for (int j = 0; j < Gold[0].length; j ++)
			{
				double gold = Gold[i][j];
				double host = Host[i][j];
				if (Math.abs(gold - host) > TOL)
				{
					PP.p("DIFF at", new FloatMat.Coord(i, j));
					PP.p("GPU =", host, "\nCPU =", gold, '\n');
					return;
				}
			}
		PP.p("PASS! ");
	}
	
	/**
	 * Check the gold standard generated from Matlab
	 */
	private static void checkGold(float[][] cpu, String goldFile)
	{
		CsvReader csv = new CsvReader("gold_" + goldFile + ".txt");
		float[][] Gold = csv.readFloatMat();
		
		for (int i = 0; i < Gold.length; i ++)
			for (int j = 0; j < Gold[0].length; j ++)
			{
				double gold = Gold[i][j];
				double host = cpu[i][j];
				if (Math.abs(gold - host) > TOL)
				{
					PP.p(goldFile, "DIFF at", new FloatMat.Coord(i, j));
					PP.p("CPU =", host, "\nMatlab =", gold, '\n');
					return;
				}
			}
		PP.p("PASS! ");
	}

	private static float[][] mult(float[][] a, float[][] b)
	{
		float[][] c = new float[a.length][b[0].length];
		for(int i=0;i<a.length;i++){
	        for(int j=0;j<b[0].length;j++){
	            for(int k=0;k<b.length;k++){
	            c[i][j]+= a[i][k] * b[k][j];
	            }
	        }
	    }
		return c;
	}
	
	private static float[][] addCol1(float[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		float[][] b = new float[r][c+1];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				b[i][j] = a[i][j];

		for (int i = 0; i < r; i++)
			b[i][c] = 1;
		
		return b;
	}
	
	private static float[][] transpose(float[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		float[][] t = new float[c][r];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				t[j][i] = a[i][j];
		return t;
	}
	
	private static void updateTheta(float[][] theta, float alpha, float[][] b, float beta)
	{
		int r = theta.length;
		int c = theta[0].length;
		
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				theta[i][j] += theta[i][j] * alpha + b[i][j] * beta;
	}
	
	private static float[][] cos(float[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				a[i][j] = (float) Math.cos(a[i][j]);
		return a;
	}
	
	private static float[][] getCol(float[][] a, int c)
	{
		int r = a.length;
		float[][] col = new float[r][1];
		for (int i = 0; i < r; i++)
			col[i][0] = a[i][c];
		return col;
	}

	private static void jbabel_id_minus_softmax(float[][] a, int id)
	{
		int r = a.length;
		float max = -Float.MAX_VALUE;
		for (int i = 0; i < r; i++)
			if (a[i][0] > max)
				max = a[i][0];
		
		for (int i = 0; i < r; i++)
			a[i][0] = (float) Math.exp( a[i][0] - max );
		
		float sum = 0;
		for (int i = 0; i < r; i++)
			sum += a[i][0];
		
		for (int i = 0; i < r; i++)
			a[i][0] *= -1.0f/sum;
		
		++ a[id][0];
	}
}
