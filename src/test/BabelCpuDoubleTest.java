package test;

import java.util.Arrays;

import gpu.*;
import utils.*;

public class BabelCpuDoubleTest
{
	private static final double TOL = 1e-5f;
	private static final int ITER = 10000;
	
	public static void main(String[] args)
	{
		PP.p("Double CPU test");
		PP.setDoublePrec(3);
		
		// Read in dummy data
		CsvReader csv = new CsvReader("input_X.txt");
		double[][] jX = csv.readDoubleMat();
		DoubleMat X = new DoubleMat(jX);
		csv = new CsvReader("input_W.txt");
		
		double[][] jW = csv.readDoubleMat();
		DoubleMat W = new DoubleMat(jW);
		csv = new CsvReader("input_Y.txt");
		int[] Y = csv.readIntVec(true);
		
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

		/*
		 * Step 1: cos(W * x + b)
		 * W and b are combined. 
		 */
		// augment X with a column of 1
		DoubleMat X1 = new DoubleMat(SAMPLES, X_DIM + 1, false);
		X1.copyFrom(X);
		ThrustNative.gpu_fill_double(X1.getThrustPointer().offset(X.size()), SAMPLES, 1);
		double[][] jX1 = addCol1(jX);
		
		// Xnew: X_NEW_DIM * SAMPLES
		DoubleMat Xnew = GpuBlas.mult(W, X1.transpose()).cos();
		double[][] jXnew = cos(mult(jW, transpose(jX1)));
//		PP.p("check Xnew"); 
//		checkGold(Xnew, jXnew);
//		checkGold(jXnew, "Xnew");
//		checkGold(Xnew, "Xnew");

		/*
		 * Step2: Create Theta matrix and compute Theta * X_new
		 */
		DoubleMat Theta = new DoubleMat(LABELS, X_NEW_DIM);
		double[][] jTheta = new double[LABELS][X_NEW_DIM];

		DoubleMat A = new DoubleMat(LABELS, 1, false);
		double[][] jA = new double[LABELS][1];
		// Loop over samples column by column
		for (int s = 0; s < ITER; ++ s)
		{
			// Step2: extract a column
			DoubleMat Xnew_s = Xnew.createOffset(s * X_NEW_DIM, X_NEW_DIM);
			double[][] jXnew_s = getCol(jXnew, s);
			

			// alpha_vector = Theta * Xnew_s, LABELS * 1
			GpuBlas.mult(Theta, Xnew_s, A);
			jA = mult(jTheta, jXnew_s);
			
			/*
			 * Step3: get Id[y==j] - P(yj | x, Theta)
			 */
			Thrust.babel_id_minus_softmax_double(A, Y[s]);
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
//		PP.p(A.sort().getHost());
//		double[] jA_ = transpose(jA)[0];  Arrays.sort(jA_);
//		PP.po(jA_);
		PP.p(A);
		PP.po(jA);
		checkGold(A, jA);
		checkGold(jA, "A");
		checkGold(A, "A");

		/*
		 * DONE!
		 * Check results against plain Java
		 */
		PP.p("Done. Check Theta:");
		checkGold(Theta, jTheta);
		checkGold(jTheta, "Theta");
		checkGold(Theta, "Theta");
		
		/*
		 * Clean up and exit
		 */
		DoubleMat[] mats = new DoubleMat[] 
				{X, W, X1, Xnew};
		for (DoubleMat mat : mats)
			mat.destroy();
		GpuBlas.destroy();
	}
	
	/**
	 * Check the gold standard from plain Java
	 */
	private static void checkGold(DoubleMat gpu, double[][] cpu)
	{
		double[][] Host = gpu.deflatten();
		
		double diff = GpuUtil.matAvgDiff(cpu, Host);
		PP.setDoublePrec(3);
		PP.setScientific(true);
		
		if (GpuUtil.matAvgDiff(cpu, Host) < TOL)
    		PP.p("PASS double GPU-CPU: ", diff);
		else
			PP.p("FAIL double GPU-CPU: ", diff);
	}
	
	/**
	 * Check the gold standard generated from Matlab
	 */
	private static void checkGold(double[][] cpu, String goldFile)
	{
		CsvReader csv = new CsvReader("gold_" + goldFile + ".txt");
		double[][] Gold = csv.readDoubleMat();
		
		double diff = GpuUtil.matAvgDiff(Gold, cpu);
		PP.setDoublePrec(3);
		PP.setScientific(true);
		
		if (GpuUtil.matAvgDiff(Gold, cpu) < TOL)
    		PP.p("PASS double CPU-Matlab: ", diff);
		else
			PP.p("FAIL double CPU-Matlab: ", diff);
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

	private static double[][] mult(double[][] a, double[][] b)
	{
		double[][] c = new double[a.length][b[0].length];
		for(int i=0;i<a.length;i++){
	        for(int j=0;j<b[0].length;j++){
	            for(int k=0;k<b.length;k++){
	            c[i][j]+= a[i][k] * b[k][j];
	            }
	        }
	    }
		return c;
	}
	
	private static double[][] addCol1(double[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		double[][] b = new double[r][c+1];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				b[i][j] = a[i][j];

		for (int i = 0; i < r; i++)
			b[i][c] = 1;
		
		return b;
	}
	
	private static double[][] transpose(double[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		double[][] t = new double[c][r];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				t[j][i] = a[i][j];
		return t;
	}
	
	private static void updateTheta(double[][] theta, double alpha, double[][] b, double beta)
	{
		int r = theta.length;
		int c = theta[0].length;
		
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				theta[i][j] += theta[i][j] * alpha + b[i][j] * beta;
	}
	
	private static double[][] cos(double[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				a[i][j] = (double) Math.cos(a[i][j]);
		return a;
	}
	
	private static double[][] getCol(double[][] a, int c)
	{
		int r = a.length;
		double[][] col = new double[r][1];
		for (int i = 0; i < r; i++)
			col[i][0] = a[i][c];
		return col;
	}
	
	private static double sum(double[][] a)
	{
		double s = 0;
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				s += a[i][j];
		return s;
	}
	
	private static void jbabel_id_minus_softmax(double[][] a, int id)
	{
		int r = a.length;
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < r; i++)
			if (a[i][0] > max)
				max = a[i][0];
		
		for (int i = 0; i < r; i++)
			a[i][0] = (double) Math.exp( a[i][0] - max );
		
		double sum = 0;
		for (int i = 0; i < r; i++)
			sum += a[i][0];
		
		for (int i = 0; i < r; i++)
			a[i][0] *= -1.0f/sum;
		
		++ a[id][0];
	}
}