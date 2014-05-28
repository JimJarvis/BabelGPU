package utils;

public class CpuUtil
{
	/**
	 * Utility: deflatten a 1D float to 2D matrix
	 * @param row row dimension
	 * @param columnMajor true for column major, false for row major
	 */
	public static float[][] deflatten(float[] A, int row, boolean columnMajor)
	{
		int col = A.length / row;
		float[][] ans = new float[row][col];
		int pt = 0;
		
		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				ans[i][j] = A[pt ++];
		else // row major
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					ans[i][j] = A[pt ++];
		return ans;
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
	
	//**************************************************/
	//******************* DOUBLE *******************/
	//**************************************************/
	/**
	 * Utility: deflatten a 1D double to 2D matrix
	 * @param row row dimension
	 * @param columnMajor true for column major, false for row major
	 */
	public static double[][] deflatten(double[] A, int row, boolean columnMajor)
	{
		int col = A.length / row;
		double[][] ans = new double[row][col];
		int pt = 0;
		
		if (columnMajor)
    		for (int j = 0; j < col; j ++)
    			for (int i = 0; i < row; i ++)
    				ans[i][j] = A[pt ++];
		else // row major
			for (int i = 0; i < row; i ++)
				for (int j = 0; j < col; j ++)
					ans[i][j] = A[pt ++];
		return ans;
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
}
