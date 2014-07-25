package utils;

import java.util.Random;

/**
 * CPU version of various matrix operations plus other stuff. 
 */
public class CpuUtil
{
	// ******************** COMMON ********************/
	private static Random rand = new Random(3760337);
	
	/**
	 * @return whether abs(actual - gold) < TOL
	 */
	public static boolean equal(double actual, double gold, double TOL)
	{
		return Math.abs(actual - gold) < TOL;
	}
	
	/**
	 * @param low inclusive
	 * @param high exclusive
	 * @return sequence of random ints
	 */
	public static int[] randInts(int size, int low, int high)
	{
		int[] rands = new int[size];
		for (int i = 0; i < rands.length; i++)
			rands[i] = rand.nextInt(high - low) + low;
		return rands;
	}

	/**
	 * @param high exclusive [0, high)
	 * @return sequence of random ints
	 */
	public static int[] randInts(int size, int high)
	{
		return randInts(size, 0, high);
	}
	
	//**************************************************/
	//******************* FLOAT *******************/
	//**************************************************/
	/**
	 * Utility: deflatten a 1D float to 2D matrix
	 * @param out output parameter
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[][] deflatten(float[] A, float[][] out, boolean columnMajor)
	{
		if (A == null)	return null;
		int row = out.length;
		int col = out[0].length;
		int pt = 0;
		
		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				out[i][j] = A[pt ++];
		else // row major
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					out[i][j] = A[pt ++];
		return out;
	}
	
	/**
	 * Utility: deflatten a 1D float to 2D matrix
	 * @param row row dimension
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[][] deflatten(float[] A, int row, boolean columnMajor)
	{
		if (A == null)	return null;
		float[][] out = new float[row][A.length / row];
		return deflatten(A, out, columnMajor);
	}
	
	/**
	 * Utility: flatten a 2D float array to 1D
	 * @param out output parameter
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[] flatten(float[][] A, float[] out, boolean columnMajor)
	{
		if (A == null)	return null;
		int row = A.length;
		int col = A[0].length;
		int pt = 0;

		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				out[pt ++] = A[i][j];
		else
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					out[pt ++] = A[i][j];

		return out;
	}
	
	/**
	 * Utility: flatten a 2D float array to 1D
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[] flatten(float[][] A, boolean columnMajor)
	{
		if (A == null)	return null;
		float[] out = new float[A.length * A[0].length];
		return flatten(A, out, columnMajor);
	}
	
	/**
	 * Returns the average of absolute difference from 2 matrices
	 */
	public static float matAvgDiff(float[][] A, float[][] B)
	{
		float diff = 0;
		int r = A.length;
		int c = A[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j ++)
				diff += Math.abs(A[i][j] - B[i][j]);
		
		return diff / (r * c);
	}
	
	public static float[][] mult(float[][] a, float[][] b)
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
	
	public static float[][] addCol1(float[][] a)
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
	
	public static float[][] transpose(float[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		float[][] t = new float[c][r];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				t[j][i] = a[i][j];
		return t;
	}
	
	public static float[][] cos(float[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				a[i][j] = (float) Math.cos(a[i][j]);
		return a;
	}
	
	public static float[][] getCol(float[][] a, int c)
	{
		int r = a.length;
		float[][] col = new float[r][1];
		for (int i = 0; i < r; i++)
			col[i][0] = a[i][c];
		return col;
	}
	
	public static float sum(float[][] a)
	{
		float s = 0;
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				s += a[i][j];
		return s;
	}
	
	public static float[][] randFloatMat(int row, int col, float low, float high)
	{
		float[][] ans = new float[row][col];
		for (int i = 0; i < row; i ++)
			for (int j = 0; j < col; j ++)
				ans[i][j] = rand.nextFloat() * (high - low) + low;
		return ans;
	}
	
	/**
	 * @return output parameter
	 */
	public static double[] float2double(float[] in, double[] out)
	{
		for (int i = 0 ; i < in.length ; i++)
			out[i] = (double) in[i];
		return out;
	}
	
	public static float[] toPrimitive(Float[] arr)
	{
		float[] ans = new float[arr.length];
		int i = 0;
		for (float x : arr)
			ans[i ++] = x;
		return ans;
	}
	
	public static Float[] toWrapper(float[] arr)
	{
		Float[] ans = new Float[arr.length];
		int i = 0;
		for (Float x : arr)
			ans[i ++] = x;
		return ans;
	}
	
	//**************************************************/
	//******************* DOUBLE *******************/
	//**************************************************/
	/**
	 * Utility: deflatten a 1D double to 2D matrix
	 * @param out output parameter
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[][] deflatten(double[] A, double[][] out, boolean columnMajor)
	{
		if (A == null)	return null;
		int row = out.length;
		int col = out[0].length;
		int pt = 0;
		
		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				out[i][j] = A[pt ++];
		else // row major
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					out[i][j] = A[pt ++];
		return out;
	}
	
	/**
	 * Utility: deflatten a 1D double to 2D matrix
	 * @param row row dimension
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[][] deflatten(double[] A, int row, boolean columnMajor)
	{
		if (A == null)	return null;
		double[][] out = new double[row][A.length / row];
		return deflatten(A, out, columnMajor);
	}
	
	/**
	 * Utility: flatten a 2D double array to 1D
	 * @param out output parameter
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[] flatten(double[][] A, double[] out, boolean columnMajor)
	{
		if (A == null)	return null;
		int row = A.length;
		int col = A[0].length;
		int pt = 0;

		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				out[pt ++] = A[i][j];
		else
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					out[pt ++] = A[i][j];

		return out;
	}
	
	/**
	 * Utility: flatten a 2D double array to 1D
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[] flatten(double[][] A, boolean columnMajor)
	{
		if (A == null)	return null;
		double[] out = new double[A.length * A[0].length];
		return flatten(A, out, columnMajor);
	}
	
	/**
	 * Returns the average of absolute difference from 2 matrices
	 */
	public static double matAvgDiff(double[][] A, double[][] B)
	{
		double diff = 0;
		int r = A.length;
		int c = A[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j ++)
				diff += Math.abs(A[i][j] - B[i][j]);
		
		return diff / (r * c);
	}
	
	public static double[][] mult(double[][] a, double[][] b)
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
	
	public static double[][] addCol1(double[][] a)
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
	
	public static double[][] transpose(double[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		double[][] t = new double[c][r];
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				t[j][i] = a[i][j];
		return t;
	}
	
	public static double[][] cos(double[][] a)
	{
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				a[i][j] = (double) Math.cos(a[i][j]);
		return a;
	}
	
	public static double[][] getCol(double[][] a, int c)
	{
		int r = a.length;
		double[][] col = new double[r][1];
		for (int i = 0; i < r; i++)
			col[i][0] = a[i][c];
		return col;
	}
	
	public static double sum(double[][] a)
	{
		double s = 0;
		int r = a.length;
		int c = a[0].length;
		for (int i = 0; i < r; i++)
			for (int j = 0; j < c; j++)
				s += a[i][j];
		return s;
	}
	
	public static double[][] randDoubleMat(int row, int col, double low, double high)
	{
		double[][] ans = new double[row][col];
		for (int i = 0; i < row; i ++)
			for (int j = 0; j < col; j ++)
				ans[i][j] = rand.nextDouble() * (high - low) + low;
		return ans;
	}
	
	/**
	 * @return output parameter
	 */
	public static float[] double2float(double[] in, float[] out)
	{
		for (int i = 0 ; i < in.length ; i++)
			out[i] = (float) in[i];
		return out;
	}
	
	public static double[] toPrimitive(Double[] arr)
	{
		double[] ans = new double[arr.length];
		int i = 0;
		for (double x : arr)
			ans[i ++] = x;
		return ans;
	}
	
	public static Double[] toWrapper(double[] arr)
	{
		Double[] ans = new Double[arr.length];
		int i = 0;
		for (Double x : arr)
			ans[i ++] = x;
		return ans;
	}
}
